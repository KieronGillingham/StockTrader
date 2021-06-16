import logging
log_template = "[%(asctime)s] %(levelname)s %(threadName)s %(name)s: %(message)s"
logging.basicConfig(level=logging.DEBUG, format=log_template, handlers= [logging.FileHandler("debug.log"), logging.StreamHandler()])
_logger = logging.getLogger(__name__)

import pandas as pd

from unittest import TestCase

from users import UserManager

import os.path as path

class TestUserManager(TestCase):

    def setUp(self):
        _logger.info(f"Setting up {__name__}")
        self.test_users_file = path.abspath("test_users.csv")

        indices = ["bobjones","charliesmith@solfintech.co.uk","megacorp@megacorp.co.uk"]
        columns = ["password", "balance"]
        data = [
            ["abc123", 100.00],
            ["cat001", 250.00],
            ["xyz321", 100000.00]
        ]
        test_users_data = pd.DataFrame(data=data, index=indices, columns=columns)
        self.user_manager = UserManager(csv_path=None)
        self.assertIsNone(self.user_manager.userbase, "Userbase created from no path.")
        self.user_manager.userbase = test_users_data
        self.assertIsNotNone(self.user_manager, "UserManager constructor failed to return instance.")
        self.assertIsNotNone(self.user_manager.userbase, "Userbase override failed.")
        self.assertIsNone(self.user_manager.current_user, "Current user is not None.")

    def test_load_userbase(self):
        _logger.info("Testing load_userbase()")
        self.user_manager = UserManager(csv_path=self.test_users_file)
        self.assertIsNotNone(self.user_manager.userbase, "Userbase failed to automatically initialise.")

        # Clear userbase to test reloading
        self.user_manager.userbase = None
        self.assertIsNone(self.user_manager.userbase, "Userbase was not cleared.")

        # Reload userbase
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
        self.assertEqual(user["balance"], 100, "Balance does not match known value.")

    def test_save_to_csv(self):
        _logger.info("Testing save_to_csv()")

        self.assertFalse(self.user_manager.save_to_csv(csv_path=None))
        self.assertFalse(self.user_manager.save_to_csv("//Invalid filepath"))

        self.assertIsNotNone(self.test_users_file)

        self.assertIsNotNone(self.user_manager.userbase)

        usercount = len(self.user_manager.userbase.index)
        self.assertTrue(self.user_manager.register_user("newuser", "newpass", 50.0))
        self.assertEqual(len(self.user_manager.userbase.index), usercount + 1)

        self.assertTrue(self.user_manager.save_to_csv(self.test_users_file))

        self.user_manager.userbase = None
        self.assertFalse(self.user_manager.save_to_csv(csv_path=self.test_users_file))

    def test_sign_out(self):
        _logger.info("Testing sign_out()")

        self.assertIsNone(self.user_manager.current_user)

        self.user_manager.current_user = self.user_manager.guest_account()

        current_user = self.user_manager.current_user

        self.assertIsNotNone(current_user)
        self.assertEqual(current_user["username"], "Guest", "Account username is incorrect.")
        self.assertEqual(current_user["password"], None, "Account password is incorrect.")
        self.assertEqual(current_user["balance"], 0, "Account balance is incorrect.")

        self.user_manager.sign_out()

        self.assertIsNone(self.user_manager.current_user)

    def test_registration(self):

        # Register valid user should succeed
        usercount = len(self.user_manager.userbase.index)
        self.assertEqual("Registration successful.", self.user_manager.register_user("newuser", "newpass", starting_balance=50.0))
        self.assertEqual(usercount + 1, len(self.user_manager.userbase.index))

        # Register user with existing username should fail
        usercount = len(self.user_manager.userbase.index)
        self.assertEqual("Username already exists.", self.user_manager.register_user("newuser", "newpass", starting_balance=50.0))
        self.assertEqual(usercount, len(self.user_manager.userbase.index))

        # Register user with no balance should default to 0
        usercount = len(self.user_manager.userbase.index)
        self.assertEqual("Registration successful.", self.user_manager.register_user("michellejones", "pass123"))
        self.assertEqual(usercount + 1, len(self.user_manager.userbase.index))
        self.assertEqual(self.user_manager.userbase[["michellejones"]]["balance"], 0)

        # Register user with no userbase set should fail

        #
        # def register_user(self, username: str, password: str, starting_balance=0):
        #     if self.userbase is None:
        #         _logger.error("Userbase not loaded. Call `load_userbase()` first.")
        #         return "User registration unavailable. User database has not been initialised."
        #     if password is None:
        #         _logger.error("Password is invalid.")
        #         return "Password is invalid."
        #     try:
        #         if username in self.userbase.index:
        #             return "Username already exists."
        #         else:
        #             user_details = {
        #                 "password": password,
        #                 "balance": starting_balance
        #             }
        #             self.userbase[username] = user_details
        #             print(self.userbase)
        #             return "Registration successful."
        #
        #     except Exception as ex:
        #         _logger.error(f"An exception occurred: {ex}")
        #         return "An exception occurred:"

    def test_guest_account(self):
        _logger.info("Testing guest_account()")
        guest_account = self.user_manager.guest_account()

        self.assertIsNotNone(guest_account, "No guest account returned.")
        self.assertEqual(guest_account["username"], "Guest", "Guest account username is not 'Guest'.")
        self.assertEqual(guest_account["password"], None, "Guest account has password.")
        self.assertEqual(guest_account["balance"], 0, "Guest account has a starting balance.")
