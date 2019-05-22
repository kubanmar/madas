import bravado, json

API_base_url = 'https://labdev-nomad.esc.rzg.mpg.de/fairdi/nomad/latest/api'
dummy_user_passwd = ('leonard.hofstadter@nomad-fairdi.tests.de', 'password')

class API():
    """
    API class to communicate with NOMAD FAIRdi website.
    """

    def __init__(self, base_url = API_base_url, username_password =  dummy_user_passwd, logger = None):
        self.username, self.password = username_password

    def get_calculation(self):
        pass

    def get_calculations_by_search(self):
        pass

    def get_property(self):
        pass
