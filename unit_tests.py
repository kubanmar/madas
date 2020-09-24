import os, sys
import pytest

if __name__ == '__main__':
    returndir = os.getcwd()
    if not os.path.exists('unit_test_data'):
        os.mkdir('unit_test_data')
    os.chdir('unit_test_data')
    path = '../simdatframe/test/'
    pytest.main([path,"-v","--cache-clear","--capture=no","--junit-xml=testlog.xml", "--fulltrace", "--cov=../simdatframe", "--cov-report=html", "--cov-config=../.coveragerc"])
    os.chdir(returndir)
