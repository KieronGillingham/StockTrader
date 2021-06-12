# Logging
import logging
_logger = logging.getLogger(__name__)

# Data manipulation
from pandas import read_csv

class UserManager:
    def __init__(self, csv_path="data/users.csv", *args, **kwargs):
        # Load data from csv by default
        self.load_userbase(csv_path=csv_path)

    def load_userbase(self, csv_path="data/users.csv"):
        _logger.debug("Loading userbase from local file")
        try:
            self.userbase = read_csv(csv_path, index_col=0)
        except FileNotFoundError as ex:
            _logger.error(f"Local storage file '{csv_path}' not found: {ex}")

    def sign_in(self, username, password):
        if self.userbase is None:
            _logger.error("Userbase not loaded. Call `load_userbase()` first.")
            return None
        try:
            user = self.userbase.loc[[username]]
            user.reset_index(inplace=True) # Move username from index to new column
            user = user.to_dict('records')[0] # Format user as a dict

            # Confirm user found and password matches
            if user is None or user["password"] != password:
                raise KeyError("Invalid username or password")

            return user
        except KeyError as ex:
            _logger.info("Invalid username or password")
            return None

    def save_to_csv(self, csv_path="data/users.csv"):
        pass
        # self.prices_df.to_csv(csv_path)

    def guest_account(self):
        return {
            "username": "Guest",
            "password": None,
            "balance": 0
        }