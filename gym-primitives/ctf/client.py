import argparse
import json
import os
import logging

logger = logging.getLogger("ctf_gym.client")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8000")
COOKIE_STORAGE_LOCATION = os.environ.get(
    "COOKIE_STORAGE_LOCATION", "/tmp/.ctf-client-cookie"
)

SSL_CERT = os.environ.get("SSL_CERT", None)

if SSL_CERT is not None:
    assert os.path.isfile(SSL_CERT), "SSL_CERT must be a file"
    logger.info("Using SSL")
    USE_SSL = True
    os.environ['REQUESTS_CA_BUNDLE'] = SSL_CERT
    logger.info("SSL_CERT: %s", SSL_CERT)
else:
    USE_SSL = False

if USE_SSL:
    SERVER_URL = SERVER_URL.replace("http://", "https://")

import requests

logger.info("SERVER_URL: %s", SERVER_URL)
logger.info("COOKIE_STORAGE_LOCATION: %s", COOKIE_STORAGE_LOCATION)


class CTFClient:
    def __init__(self, cookie_storage_location: str = COOKIE_STORAGE_LOCATION):
        self.cookie_storage_location = cookie_storage_location
        if os.path.exists(cookie_storage_location):
            with open(cookie_storage_location, "r") as f:
                self.session_cookie = f.read().strip()
        else:
            self.session_cookie = None

    def register(self, username: str):
        if self.session_cookie is not None:
            return "Already registered"
        response = requests.post(f"{SERVER_URL}/register/{username}")
        if response.status_code == 200:
            self.session_cookie = response.json()["cookie"]
            with open(self.cookie_storage_location, "w") as f:
                f.write(self.session_cookie)
            return f"Registered as {username}. Cookie: {self.session_cookie}"
        return "Registration failed"

    def list_challenges(self):
        response = requests.get(f"{SERVER_URL}/challenges", cookies=self.cookies)
        return response.json()

    def list_all_challenges(self):
        response = requests.get(
            f"{SERVER_URL}/challenges",
            params=dict(show_solved=True, show_locked=True),
            cookies=self.cookies,
        )
        return response.json()

    def get_challenge(self, challenge_id: str):
        response = requests.get(
            f"{SERVER_URL}/challenge/{challenge_id}", cookies=self.cookies
        )
        return response.json()

    def submit_solution(self, challenge_id: str, solution: str):
        response = requests.post(
            f"{SERVER_URL}/challenge/{challenge_id}/submit",
            params={"solution": solution},
            cookies=self.cookies,
        )
        return response.json()

    @property
    def cookies(self):
        return {"session": self.session_cookie} if self.session_cookie else {}

    def get_scoreboard(self):
        response = requests.get(f"{SERVER_URL}/scoreboard")
        return response.json()

    def whoami(self):
        response = requests.get(f"{SERVER_URL}/whoami", cookies=self.cookies)
        return response.json()

    def info(self):
        response = requests.get(f"{SERVER_URL}/info", cookies=self.cookies)
        return response.json()

    def score(self):
        response = requests.get(f"{SERVER_URL}/score", cookies=self.cookies)
        return response.json()

    def get_submission_limit(self):
        response = requests.get(f"{SERVER_URL}/submission_limit", cookies=self.cookies)
        return response.json()

    def download(self, filename: str):
        # first check if the server has file hosted
        if self.info()["has_files"] == False:
            return "Server does not have files hosted"
        url = f"{SERVER_URL}/static/{filename}"
        savename = os.path.basename(filename)
        if os.path.exists(savename):
            ans = input("File '%s' already exists, overwrite? (y/n) " % savename)
            if ans.lower() != "y":
                return "Not overwriting"
        response = requests.get(url, cookies=self.cookies)
        if response.status_code != 200:
            return f"Failed to download file '{filename}'"
        with open(savename, "wb") as f:
            f.write(response.content)
        return f"Downloaded file '{filename}' to '{savename}'"


def main():
    # TODO: print the output in yaml, not json, since we have newline in string
    client = CTFClient()
    parser = argparse.ArgumentParser(description="CTF Client")
    subparsers = parser.add_subparsers(dest="command")

    # Register command
    register_parser = subparsers.add_parser("register")
    register_parser.add_argument("username")

    # List challenges command
    subparsers.add_parser("list")

    # List all challenges, including solved ones
    subparsers.add_parser("list_all")

    # Get challenge command
    get_parser = subparsers.add_parser("get")
    get_parser.add_argument("challenge_id")

    # Submit command
    submit_parser = subparsers.add_parser("submit")
    submit_parser.add_argument("challenge_id")
    submit_parser.add_argument("solution")

    # Scoreboard command
    subparsers.add_parser("scoreboard")

    # Whoami command
    subparsers.add_parser("whoami")

    # Download command
    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("filename")

    # Info command
    subparsers.add_parser("info")

    # Score command
    subparsers.add_parser("score")

    # Submission count command
    subparsers.add_parser("submission_limit")

    args = parser.parse_args()

    if args.command == "register":
        print(client.register(args.username))
    elif args.command == "list":
        print(json.dumps(client.list_challenges(), indent=2))
    elif args.command == "list_all":
        print(json.dumps(client.list_all_challenges(), indent=2))
    elif args.command == "get":
        print(json.dumps(client.get_challenge(args.challenge_id), indent=2))
    elif args.command == "submit":
        print(
            json.dumps(
                client.submit_solution(args.challenge_id, args.solution), indent=2
            )
        )
    elif args.command == "submission_limit":
        print(json.dumps(client.get_submission_limit(), indent=2))
    elif args.command == "scoreboard":
        print(json.dumps(client.get_scoreboard(), indent=2))
    elif args.command == "whoami":
        print(json.dumps(client.whoami(), indent=2))
    elif args.command == "info":
        print(json.dumps(client.info(), indent=2))
    elif args.command == "score":
        print(json.dumps(client.score(), indent=2))
    elif args.command == "download":
        print(client.download(args.filename))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
