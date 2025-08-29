import uuid
import yaml
import os
from typing import Set, Optional
from fastapi import FastAPI, Cookie, Response, BackgroundTasks
import uvicorn
import tinydb
from threading import Lock
from pydantic import BaseModel, model_validator
from fastapi.staticfiles import StaticFiles
import logging

logger = logging.getLogger("ctf_gym.server")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# TODO: hide solved challenges, by parameters
# TODO: fix whoami

# TODO: add validator detailed error, showing detail debug info

app = FastAPI()

CHALLENGE_LOADPATH = os.environ.get("CHALLENGE_LOADPATH", "./challenges.yaml")

CHALLENGE_LOADPATH = os.path.abspath(CHALLENGE_LOADPATH)

TINYDB_PERSIST_PATH = os.environ.get("TINYDB_PERSIST_PATH", "./ctf-server-data.json")

USE_SSL = os.environ.get("USE_SSL", "0") == "1"

SSL_CERT = os.environ.get("SSL_CERT", None)
SSL_KEY = os.environ.get("SSL_KEY", None)

if USE_SSL:
    logger.info("Using SSL")
    ssl_cert = SSL_CERT
    ssl_key = SSL_KEY
    if ssl_cert is None or ssl_key is None:
        raise ValueError("SSL_CERT and SSL_KEY must be set")
    assert os.path.isfile(ssl_cert), f"SSL_CERT {ssl_cert} not found"
    assert os.path.isfile(ssl_key), f"SSL_KEY {ssl_key} not found"
    logger.info("SSL_CERT: %s", ssl_cert)
    logger.info("SSL_KEY: %s", ssl_key)

# print all configurations
logger.info("CHALLENGE_LOADPATH: %s", CHALLENGE_LOADPATH)
logger.info("TINYDB_PERSIST_PATH: %s", TINYDB_PERSIST_PATH)

assert os.path.isfile(
    CHALLENGE_LOADPATH
), f"Challenge file {CHALLENGE_LOADPATH} not found"


class CTFChallenge(BaseModel):
    id: str
    title: str
    description: str
    points: int
    solution: str
    unlocks: list[str] = []
    locked: bool = False

    @model_validator(mode="after")
    def validate_points(self):
        assert self.points > 0, "Points must be greater than 0, given %s" % self.points
        return self

    @model_validator(mode="after")
    def validate_title(self):
        assert self.title, "Title must be set"
        return self

    @model_validator(mode="after")
    def validate_description(self):
        assert self.description, "Description must be set"
        return self

    @model_validator(mode="after")
    def validate_solution(self):
        assert self.solution, "Solution must be set"
        return self

    @model_validator(mode="after")
    def validate_id(self):
        assert self.id, "ID must be set"
        return self


class CTFChallengeCollection(BaseModel):
    name: str
    description: str
    challenges: list[CTFChallenge]
    solution_prefix: str
    solution_suffix: str
    solution_max_length: int
    solution_min_length: int
    challenge_min_points: int
    challenge_max_points: int
    solution_format_description: str
    file_loadpath: Optional[str] = None
    max_submissions: int

    # TODO: Implement challenge min/max points verifier

    @model_validator(mode="after")
    def validate_max_submissions(self):
        """
        Validate max submissions to be greater than 0.

        Raises:
            ValueError: If max submissions is not greater than 0.
        """
        assert self.max_submissions > 0, (
            "Max submissions must be greater than 0, given %s" % self.max_submissions
        )
        return self

    @model_validator(mode="after")
    def validate_description(self):
        assert self.description, "Description must be set"
        return self

    @model_validator(mode="after")
    def validate_solution_format_description(self):
        assert (
            self.solution_format_description
        ), "Solution format description must be set"
        return self

    # validate the id is unique
    @model_validator(mode="after")
    def validate_unique_id(self):
        ids = [challenge.id for challenge in self.challenges]
        for it in ids:
            count = ids.count(it)
            if count > 1:
                raise ValueError(
                    "Challenge ids must be unique, given duplicate id '%s' (total %s times)"
                    % (it, count)
                )
        return self

    @model_validator(mode="after")
    def validate_solution_prefix(self):
        assert self.solution_prefix
        solutions = [challenge.solution for challenge in self.challenges]
        for solution in solutions:
            if not solution.startswith(self.solution_prefix):
                raise ValueError(
                    "Solution must start with solution_prefix '%s', given '%s'"
                    % (self.solution_prefix, solution)
                )
        return self

    @model_validator(mode="after")
    def validate_solution_suffix(self):
        assert self.solution_suffix
        solutions = [challenge.solution for challenge in self.challenges]
        for solution in solutions:
            if not solution.endswith(self.solution_suffix):
                raise ValueError(
                    "Solution must end with solution_suffix '%s', given '%s'"
                    % (self.solution_suffix, solution)
                )
        return self

    @model_validator(mode="after")
    def validate_challenge_points(self):
        assert self.challenge_min_points > 0, (
            "challenge_min_points must be longer than 0, given %s"
            % self.challenge_min_points
        )
        assert self.challenge_max_points > 0, (
            "challenge_max_points must be longer than 0, given %s"
            % self.challenge_max_points
        )
        assert self.challenge_min_points <= self.challenge_max_points, (
            "challenge_max_points must be greater or equal to challenge_min_points, given challenge_max_points %s and challenge_min_points %s"
            % (self.challenge_max_points, self.challenge_min_points)
        )
        points_list = [challenge.points for challenge in self.challenges]
        for points in points_list:
            if not points <= self.challenge_max_points:
                raise ValueError(
                    "Points must be less or equal to  challenge_max_points %s, given %s"
                    % (self.challenge_max_points, points)
                )
            if not points >= self.challenge_min_points:
                raise ValueError(
                    "Points must be greater or equal to challenge_min_points %s, given %s"
                    % (self.challenge_min_points, points)
                )
        return self

    @model_validator(mode="after")
    def validate_solution_length(self):
        assert self.solution_min_length > 0, (
            "solution_min_length must be longer than 0, given %s"
            % self.solution_min_length
        )
        assert self.solution_max_length > 0, (
            "solution_max_length must be longer than 0, given %s"
            % self.solution_max_length
        )
        assert self.solution_min_length <= self.solution_max_length, (
            "solution_max_length must be greater or equal to solution_min_length, given solution_max_length %s and solution_min_length %s"
            % (self.solution_max_length, self.solution_min_length)
        )
        solutions = [challenge.solution for challenge in self.challenges]
        for solution in solutions:
            if not len(solution) <= self.solution_max_length:
                raise ValueError(
                    "Solution must be less or equal to length solution_max_length %s, given '%s' (length: %s)"
                    % (self.solution_max_length, solution, len(solution))
                )
            if not len(solution) >= self.solution_min_length:
                raise ValueError(
                    "Solution must be greater or equal to length solution_min_length %s, given '%s' (length: %s)"
                    % (self.solution_min_length, solution, len(solution))
                )
        return self


# Load challenges
with open(CHALLENGE_LOADPATH, "r") as f:
    challenge_collection: dict = yaml.safe_load(f)
    challenge_collection = CTFChallengeCollection.model_validate(
        challenge_collection
    ).model_dump()

challenges = challenge_collection["challenges"]
logger.info("Challenge collection: %s", challenge_collection["name"])
logger.info("Loaded challenges: %s", len(challenges))

COLLECTION_NAME = challenge_collection["name"]
COLLECTION_DESCRIPTION = challenge_collection["description"]

CHALLENGE_COUNT = len(challenges)

FILE_LOADPATH = challenge_collection.get("file_loadpath", None)

SOLUTION_MAX_LENGTH = challenge_collection["solution_max_length"]
SOLUTION_MIN_LENGTH = challenge_collection["solution_min_length"]
SOLUTION_PREFIX = challenge_collection["solution_prefix"]
SOLUTION_SUFFIX = challenge_collection["solution_suffix"]
SOLUTION_FORMAT_DESCRIPTION = challenge_collection["solution_format_description"]
SUBMISSION_LIMIT = challenge_collection["max_submissions"]

COLLECTION_HAS_FILES = False

if FILE_LOADPATH:
    FILE_LOADPATH = os.path.abspath(FILE_LOADPATH)
    assert os.path.dirname(CHALLENGE_LOADPATH) in os.path.dirname(FILE_LOADPATH), (
        "File loadpath must be a subdirectory of challenge file directory, given %s (challenge file directory) and %s (file loadpath)"
        % (CHALLENGE_LOADPATH, FILE_LOADPATH)
    )
    if os.path.isdir(FILE_LOADPATH):
        COLLECTION_HAS_FILES = True
        app.mount("/static", StaticFiles(directory=FILE_LOADPATH), name="static")
    else:
        raise ValueError(f"File loadpath {FILE_LOADPATH} is not a directory")

# Initialize TinyDB
db = tinydb.TinyDB(TINYDB_PERSIST_PATH)
sessions_table = db.table("sessions")
solved_table = db.table("solved_challenges")
scores_table = db.table("scores")
unlocked_table = db.table("unlocked_challenges")
# use a global counter for submission count
submission_count_table = db.table("submission_count")
# Thread lock for write operations
db_lock = Lock()

# Load existing data from TinyDB
sessions = {doc["cookie"]: doc["username"] for doc in sessions_table.all()}
solved_challenges = {
    doc["username"]: set(doc["challenges"]) for doc in solved_table.all()
}
scores = {doc["username"]: doc["score"] for doc in scores_table.all()}
unlocked_challenges = {
    doc["username"]: set(doc["challenges"]) for doc in unlocked_table.all()
}
submission_count = {
    doc["username"]: doc["count"] for doc in submission_count_table.all()
}


# Database operation functions
def persist_session(cookie: str, username: str):
    with db_lock:
        sessions_table.insert({"cookie": cookie, "username": username})


def persist_submission_count(username: str, count: int):
    with db_lock:
        submission_count_table.upsert(
            {"username": username, "count": count},
            tinydb.Query().username == username,
        )


def persist_unlocked_challenges(username: str, challenges: Set[str]):
    with db_lock:
        unlocked_table.upsert(
            {"username": username, "challenges": list(challenges)},
            tinydb.Query().username == username,
        )


def persist_solved_challenges(username: str, challenges: Set[str]):
    with db_lock:
        solved_table.upsert(
            {"username": username, "challenges": list(challenges)},
            tinydb.Query().username == username,
        )


def persist_score(username: str, score: int):
    with db_lock:
        scores_table.upsert(
            {"username": username, "score": score}, tinydb.Query().username == username
        )


def get_username_from_cookie(cookie: str) -> str:
    return sessions.get(cookie, "")


@app.get("/whoami")
def whoami(session: str = Cookie(None)):
    status = "not_registered"
    username = get_username_from_cookie(session)
    if username:
        status = "ok"
    return {
        "status": status,
        "data": {"username": username},
    }


@app.get("/submission_limit")
def get_submission_limit(session: str = Cookie(None)):
    username = get_username_from_cookie(session)
    if not username:
        return {"error": "Not authenticated"}
    total_submissions = submission_count[username]
    threshold_hit = total_submissions >= SUBMISSION_LIMIT
    return {
        "total_submissions": total_submissions,
        "threshold_hit": threshold_hit,
        "submission_limit": SUBMISSION_LIMIT,
    }


@app.post("/register/{username}")
def register(username: str, response: Response, background_tasks: BackgroundTasks):
    assert username, "Username cannot be empty"
    if username in sessions.values():
        response.status_code = 400
        return {"message": "Username already registered"}

    cookie = str(uuid.uuid4())
    # Update in-memory storage
    sessions[cookie] = username
    solved_challenges[username] = set()
    unlocked_challenges[username] = set()
    scores[username] = 0
    submission_count[username] = 0

    # Add background tasks for persistence
    background_tasks.add_task(persist_session, cookie, username)
    background_tasks.add_task(persist_solved_challenges, username, set())
    background_tasks.add_task(persist_unlocked_challenges, username, set())
    background_tasks.add_task(persist_submission_count, username, 0)
    background_tasks.add_task(persist_score, username, 0)

    response.set_cookie(key="session", value=cookie)
    return {"message": "Registered successfully", "cookie": cookie}


@app.get("/challenges")
def list_challenges(
    response: Response,
    session: str = Cookie(None),
    show_solved: bool = False,
    show_locked: bool = False,
):
    """
    Return a list of challenges the user can attempt.

    Args:
        session (str): The session cookie, used to authenticate the user.
        show_solved (bool): If True, show challenges that the user has already solved.
        show_locked (bool): If True, show challenges that the user has not unlocked yet.
    Returns:
        List of dictionaries with challenge id and title.
    """
    username = get_username_from_cookie(session)
    if not username:
        response.status_code = 401
        return {"message": "Unauthorized"}
    ret = [
        {
            "id": ch["id"],
            "title": ch["title"],
            "points": ch["points"],
            "solved": solved_challenges[username].__contains__(ch["id"]),
            "unlocked": (not ch["locked"])
            or unlocked_challenges[username].__contains__(ch["id"]),
        }
        for ch in challenges
    ]
    if not show_solved:
        ret = [ch for ch in ret if not ch["solved"]]
    if not show_locked:
        ret = [ch for ch in ret if ch["unlocked"]]
    return ret


@app.get("/challenge/{challenge_id}")
def get_challenge(challenge_id: str, session: str = Cookie(None)):
    username = get_username_from_cookie(session)
    if not username:
        return {"error": "Not authenticated"}

    for challenge in challenges:
        if challenge["id"] == challenge_id:
            solved = challenge["id"] in solved_challenges[username]
            unlocked = (not challenge["locked"]) or challenge[
                "id"
            ] in unlocked_challenges[username]
            if not unlocked:
                return {"error": "Challenge not unlocked"}
            return {
                "title": challenge["title"],
                "description": challenge["description"],
                "points": challenge["points"],
                "solved": solved,
            }
    return {"error": "Challenge not found"}


def validate_solution_format(solution: str) -> bool:
    if not solution.startswith(SOLUTION_PREFIX):
        return False
    if not solution.endswith(SOLUTION_SUFFIX):
        return False
    if len(solution) > SOLUTION_MAX_LENGTH:
        return False
    if len(solution) < SOLUTION_MIN_LENGTH:
        return False
    return True


@app.post("/challenge/{challenge_id}/submit")
def submit_solution(
    challenge_id: str,
    solution: str,
    session: str = Cookie(None),
    background_tasks: BackgroundTasks = None,
):
    username = get_username_from_cookie(session)
    if not username:
        return {"error": "Not authenticated"}

    for challenge in challenges:
        if challenge["id"] == challenge_id:
            unlocked = (not challenge["locked"]) or challenge_id in unlocked_challenges[
                username
            ]
            if not unlocked:
                return {"error": "Challenge not unlocked"}
            if not validate_solution_format(solution):
                return {
                    "error": "Solution format invalid",
                    "message": f"Solution must start with '{SOLUTION_PREFIX}', end with '{SOLUTION_SUFFIX}', and be at most {SOLUTION_MAX_LENGTH} characters long, at least {SOLUTION_MIN_LENGTH} characters long",
                }
            # now the solution is valid

            # check if the user has ran out of submissions
            if submission_count[username] >= SUBMISSION_LIMIT:
                return {"error": "Submission limit reached"}

            # increment the number of submission
            submission_count[username] += 1
            background_tasks.add_task(
                persist_submission_count, username, submission_count[username]
            )

            # check if it is correct
            if solution == challenge["solution"]:
                if challenge_id not in solved_challenges[username]:
                    # Update in-memory storage
                    solved_challenges[username].add(challenge_id)
                    scores[username] += challenge.get("points", 1)

                    # Add background tasks for persistence
                    background_tasks.add_task(
                        persist_solved_challenges, username, solved_challenges[username]
                    )
                    background_tasks.add_task(persist_score, username, scores[username])
                else:
                    return {"error": "Challenge already solved"}
                # add unlocked challenges
                if "unlocks" in challenge:
                    for unlock_challenge_id in challenge["unlocks"]:
                        if unlock_challenge_id not in unlocked_challenges[username]:
                            unlocked_challenges[username].add(unlock_challenge_id)
                            background_tasks.add_task(
                                persist_unlocked_challenges,
                                username,
                                unlocked_challenges[username],
                            )
                return {"correct": True, "message": "Correct solution!"}
            else:
                return {"correct": False, "message": "Incorrect solution"}
    return {"error": "Challenge not found"}


@app.get("/scoreboard")
def get_scoreboard():
    return sorted(
        [{"username": u, "score": s} for u, s in scores.items()],
        key=lambda x: x["score"],
        reverse=True,
    )


@app.get("/info")
def get_info():
    return {
        "name": COLLECTION_NAME,
        "description": COLLECTION_DESCRIPTION,
        "challenge_count": CHALLENGE_COUNT,
        "solution_max_length": SOLUTION_MAX_LENGTH,
        "solution_min_length": SOLUTION_MIN_LENGTH,
        "solution_prefix": SOLUTION_PREFIX,
        "solution_suffix": SOLUTION_SUFFIX,
        "solution_format_description": SOLUTION_FORMAT_DESCRIPTION,
        "has_files": COLLECTION_HAS_FILES,
        "submission_limit": SUBMISSION_LIMIT,
    }


@app.get("/score")
def get_score(session: str = Cookie(None)):
    username = get_username_from_cookie(session)
    if not username:
        return {"error": "Not authenticated"}
    return {"score": scores[username]}


def main():
    if USE_SSL:
        uvicorn.run(
            app, host="0.0.0.0", port=8000, ssl_keyfile=SSL_KEY, ssl_certfile=SSL_CERT
        )
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
