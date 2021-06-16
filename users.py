# Logging
import logging
_logger = logging.getLogger(__name__)

# Data manipulation
from pandas import read_csv

class UserManager:
    """
    A simple manager for user accounts.
    WARNING: This implementation is insecure and should not be used in production environments.
    """

    def __init__(self, csv_path="data/users.csv"):
        """
        Initalise a UserManager instance.
        :param csv_path: The path to use for storing the user csv file.
        """
        # Set default values
        self.userbase = None
        self.current_user = None

        if csv_path is not None:
            self.load_userbase(csv_path=csv_path)

    def load_userbase(self, csv_path="data/users.csv"):
        """
        Load a local csv file to get user information.
        This implementation uses unsecured csv files containing plaintext passwords.
        :param csv_path: The path of the user csv file to load.
        :return: None.
        """
        # Load from given file
        if csv_path is not None:
            _logger.debug("Loading userbase from local file.")
            try:
                # Read csv file using first column (username) as the index
                userbase = read_csv(csv_path, index_col=0)
                # Check userbase was successfully loaded
                if userbase is None:
                    raise Exception(f"Userbase {csv_path} not successfully read.")
                # Set userbase as global reference
                self.userbase = userbase
            except FileNotFoundError as ex:
                _logger.error(f"Local storage file '{csv_path}' not found: {ex}")
            except Exception as ex:
                _logger.error(f"An exception occurred: {ex}")
        else:
            _logger.error(f"No local storage file specified.")

    def sign_in(self, username, password):
        """
        Sign in a user. This will set the current_user attribute to a dictionary containing the user's details.
        :param username: The username of the user being signed in.
        :param password: The password entered to sign in.
        :return: A dictionary of the current user's details if successfully verified. Otherwise, None.
        """
        # Check userbase exists
        if self.userbase is None:
            _logger.error("Userbase not loaded. Call `load_userbase()` first.")
            return None
        try:
            # Select user row by (unique) username
            user = self.userbase.loc[[username]]

            # Check if user is found
            if user is None:
                raise KeyError()

            # Format user details as a dict
            user.reset_index(inplace=True) # Move username from index to new column
            records = user.to_dict('records')
            if isinstance(records, list): # Confirm single element list containing a dict is returned
                if len(records) != 1:
                    raise Exception(f"{len(records)} records found matching username '{username}'.")
                # Get first user record
                record = records[0]

                if not isinstance(record, dict): # Confirm user record is a dict
                    raise Exception(f"{record} is not a dict.")
            else:
                raise Exception(f"{records} is not a list of users.")

            # Confirm a user has been found
            if record is not None:
                # Confirm password matches
                if record["password"] != password:
                    raise KeyError()

                # Sign out current user if any
                if self.current_user is not None:
                    self.sign_out()

                # Set new user as current
                del record["password"] # Remove password before returning dict
                self.current_user = record
                return self.current_user

        except KeyError as ex:
            _logger.info("Invalid username or password.")
            return None
        except Exception as ex:
            _logger.error(f"An exception occurred: {ex}")

    def sign_out(self):
        """
        Sign out the current user.
        :return: None.
        """
        if self.current_user is not None:
            self.current_user = None
        else:
            _logger.warning("No current user to sign out.")

    def save_to_csv(self, csv_path="data/users.csv"):
        """
        Save out userbase to a csv file.
        :param csv_path: The path of the new csv file.
        :return: True if successful. False otherwise.
        """
        try:
            if csv_path is None:
                raise Exception("No local storage file specified.")

            elif self.userbase is None:
                raise Exception("No user data to save.")

            # Save to given file
            self.userbase.to_csv(csv_path)
            return True

        except FileNotFoundError as ex:
            _logger.error(f"Local storage file '{csv_path}' not found: {ex}")
        except Exception as ex:
            _logger.error(f"An exception occurred: {ex}")

        # Save failed
        return False

    def register_user(self, username : str, password : str, starting_balance=0):
        if self.userbase is None:
            _logger.error("Userbase not loaded. Call `load_userbase()` first.")
            return "User registration unavailable. User database has not been initialised."
        if password is None:
            _logger.error("Password is invalid.")
            return "Password is invalid."
        try:
            if username in self.userbase.index:
                return "Username already exists."
            else:
                user_details = {
                    "password": password,
                    "balance": starting_balance
                }
                self.userbase.loc[username] = user_details
                return "Registration successful."

        except Exception as ex:
            _logger.error(f"An exception occurred: {ex}")
            return "An exception occurred."

    def guest_account(self):
        """
        Return a blank guest account with default credentials.
        :return: A user account dict containing credentials for a guest user.
        """
        return {
            "username": "Guest",
            "password": None,
            "balance": 0
        }