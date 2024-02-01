import pytest

from madas.utils import BatchIterator, print_dict_tree, print_key_paths

@pytest.fixture
def example_dict():
    return {
        "a" : {
            "b" : 2, 
            "c" : [
                {"d" : 3},
                {"e" : 4},
                5,
                "f"
            ]
        },
        "g" : [1, 2, 3]
    }


def test_BatchIterator_init():

    with pytest.raises(ValueError):
        _ = BatchIterator(4, 2, task_id=1)

def test_BatchIterator_get_batches_for_index():
    
    bi = BatchIterator(11, 3)

    ref_index = 3

    batches_with_ref_index = [
        [[0, 3], [3, 6]], 
        [[3, 6], [3, 6]], 
        [[3, 6], [6, 9]], 
        [[3, 6], [9, 11]], 
    ]

    assert bi.get_batches_for_index(ref_index) == batches_with_ref_index, "Did not return correct batches for index"

    bi = BatchIterator(11, 3, symmetric=False)

    ref_index = 3

    batches_with_ref_index = [
        [[0, 3], [3, 6]], 
        [[3, 6], [3, 6]], 
        [[6, 9], [3, 6]], 
        [[9, 11], [3, 6]], 
    ]

    assert bi.get_batches_for_index(ref_index) == batches_with_ref_index, "Did not return correct batches for index for asymmetric case"

def test_BatchIterator_get_batch_rows():

    bi_t1 = BatchIterator(8,3, n_tasks=2, task_id=0)
    bi_t2 = BatchIterator(8,3, n_tasks=2, task_id=1)

    rows_t1 = [
        [
            [[0, 3], [0, 3]], 
            [[0, 3], [3, 6]], 
            [[0, 3], [6, 8]]
        ], 
        [
            [[0, 3], [6, 8]], 
            [[3, 6], [6, 8]], 
            [[6, 8], [6, 8]]]
    ]

    rows_t2 = [
        [
            [[0, 3], [3, 6]], 
            [[3, 6], [3, 6]], 
            [[3, 6], [6, 8]]
        ]
    ]

    assert bi_t1.get_batch_rows() == rows_t1, "Wrong batch rows for task 0"
    assert bi_t2.get_batch_rows() == rows_t2, "Wrong batch rows for task 1"

def test_BatchIterator_iter():

    size = 11
    n_tasks = 3
    batch_size = 2
    for task_id in range(n_tasks):
        bi = BatchIterator(size,batch_size,task_id=task_id, n_tasks = n_tasks, symmetric=True)
        batches_split = list(bi)
        assert batches_split == bi.batches[task_id::n_tasks], f"Task {task_id} did not return correct batches"

def test_BatchIterator_linear_batch_list():

    linear_list = BatchIterator.linear_batch_list(5,2)

    expected_list = [
        [0,2],
        [2,4],
        [4,5]
    ]

    assert linear_list == expected_list, "Wrong linear list"

def test_print_dict_tree(example_dict, capsys):
    print_dict_tree(example_dict)
    out, err = capsys.readouterr()
    assert len(err) == 0, f"Produced an error: {err}"
    assert out == """ a {
   b : <value>
   c [ 
   [0] {
     d : <value>
  }
   [1] {
     e : <value>
  }
     <value>
  ]
}
 g : <value>
""", "Did not print correct tree!"
    
def test_print_key_paths(example_dict, capsys):
    print_key_paths("e", example_dict)
    out, err = capsys.readouterr()
    assert len(err) == 0, f"Produced an error: {err}"
    assert out == """/a/c/1/e\n""", "Did not print correct path!"