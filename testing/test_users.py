import logging
log_template = "[%(asctime)s] %(levelname)s %(threadName)s %(name)s: %(message)s"
logging.basicConfig(level=logging.DEBUG, format=log_template, handlers= [logging.FileHandler("debug.log"), logging.StreamHandler()])
_logger = logging.getLogger(__name__)

from unittest import TestCase

from users import UserManager

import os.path as path

class TestUserManager(TestCase):

    def setUp(self):
        _logger.info(f"Setting up {__name__}")
        self.test_users_file = path.abspath("test_users.csv")
        self.user_manager = UserManager(csv_path=self.test_users_file)
        self.assertIsNotNone(self.user_manager, "UserManager constructor failed to return instance.")

    def test_load_userbase(self):
        _logger.info("Testing load_userbase()")
        self.assertIsNotNone(self.user_manager.userbase, "Userbase failed to automatically initialise.")

        # Clear userbase to test reloading
        self.user_manager.userbase = None
        self.assertIsNone(self.user_manager.userbase, "Userbase was not cleared.")

        self.user_manager.load_userbase(self.test_users_file)
        self.assertIsNotNone(self.user_manager.userbase, "Userbase failed to be reloaded.")

    def test_sign_in(self):
        _logger.info("Testing sign_in()")

        realuser = "bobjones"
        realpass = "abc123"

        user = self.user_manager.sign_in(None, None)
        self.assertIsNone(user, "User returned without any credentials.")

        user = self.user_manager.sign_in(realuser, None)
        self.assertIsNone(user, "User returned without password.")

        user = self.user_manager.sign_in(None, realpass)
        self.assertIsNone(user, "User returned without username.")

        user = self.user_manager.sign_in("fakeuser", "fakepass")
        self.assertIsNone(user, "User returned with fake credentials.")

        user = self.user_manager.sign_in(realuser, realpass)
        self.assertEqual(user["username"], realuser, "Username does not match given one.")
        self.assertEqual(user["password"], realpass, "Password does not match given one.")
        self.assertEqual(user["balance"], 100, "Balance does not match known value.")

    def test_save_to_csv(self):
        _logger.info("Testing save_to_csv()")
        self.fail()

    def test_guest_account(self):
        _logger.info("Testing guest_account()")
        guest_account = self.user_manager.guest_account()

        self.assertIsNotNone(guest_account, "No guest account returned.")
        self.assertEqual(guest_account["username"], "Guest", "Guest account username is not 'Guest'.")
        self.assertEqual(guest_account["password"], None, "Guest account has password.")
        self.assertEqual(guest_account["balance"], 0, "Guest account has a starting balance.")
