# TODO: split execution logs with commandline

import os
import sys
from log_utils import logger_print
import shutil
import subprocess

# TODO: backup .gitconfig file in user directory, if config related command fails we restore it and rerun the command
# TODO: recursively backup all .git folders in submodules
# TODO: repair corrupted index by renaming index file and `git reset`

from enum import auto
from strenum import StrEnum
REQUIRED_BINARIES = [RCLONE := "rclone", GIT := "git"]

DISABLE_GIT_AUTOCRLF = f'{GIT} config --global core.autocrlf input'
PRUNE_NOW = f'{GIT} gc --prune=now'
SCRIPT_FILENAME = os.path.basename(__file__)

# TODO: combine this with other git config commands
# os.system(DISABLE_GIT_AUTOCRLF)

# import parse
from config_utils import EnvBaseModel, getConfig
import filelock
import pathlib

# TODO: automatic configure git safe directory
# TODO: use builtin atomic feature of git, or even better, just get latest backup from remote (you may still need to backup config and hooks locally)
# ref: https://git-scm.com/search/results?search=atomic


class BackupUpdateCheckMode(StrEnum):
    commit_and_backup_flag_metadata = auto()
    git_commit_hash = auto()

USER_HOME = os.path.expanduser("~")

GIT_CONFIG_FNAME = ".gitconfig"
GIT_CONFIG_BACKUP_FNAME = ".gitconfig.bak"
GIT_CONFIG_ABSPATH = os.path.join(USER_HOME, GIT_CONFIG_FNAME)
GIT_CONFIG_BACKUP_ABSPATH = os.path.join(USER_HOME, GIT_CONFIG_BACKUP_FNAME)
GIT_LIST_CONFIG = f"{GIT} config -l"
GIT_ADD_GLOBAL_CONFIG_CMDGEN = (
    lambda conf: f"{GIT} config --global --add {conf.split('=')[0]} \"{conf.split('=')[1]}\""
)


def add_safe_directory():
    """
    We do this here so you don't have to.
    """
    success = False
    p = subprocess.run(GIT_LIST_CONFIG.split(), stdout=subprocess.PIPE)
    curdir = os.path.abspath(".").replace("\\", "/")
    assert (
        p.returncode == 0
    ), f"Abnormal return code {p.returncode} while listing git configuration"
    target_conf = f"safe.directory={curdir}"
    if target_conf not in p.stdout.decode("utf-8"):
        cmd = GIT_ADD_GLOBAL_CONFIG_CMDGEN(target_conf)
        logger_print("Running command:", cmd)
        return_code = os.system(cmd)
        assert (
            return_code == 0
        ), f"Abnormal return code {return_code} while adding current directory to safe directories"
    success = True
    return success


def exec_system_command_and_check_return_code(command:str, banner:str):
    success = False
    ret = os.system(command)
    success = ret == 0
    assert success, f"{banner.title()} command failed with exit code {ret}"
    return success


def detect_upstream_branch():
    try:
        upstream = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', '@{upstream}'], stderr=subprocess.STDOUT).decode().strip()
        logger_print("Upstream branch: " + upstream)
        return upstream
    except subprocess.CalledProcessError:
        raise Exception("Error: Current branch has no upstream branch set\nHint: git branch --set-upstream-to=<origin name>/<branch> <current branch>")

def detect_upstream_branch_and_add_safe_directory():
    success = False
    # do not swap the order.
    success = add_safe_directory()
    if not success:
        # repair
        logger_print("repairing .gitconfig")
        repaired = restore_gitconfig()
        if repaired:
            success = add_safe_directory()
    if success:
        # backup
        logger_print("backing up .gitconfig")
        backedUp = backup_gitconfig()
        if backedUp:
            detect_upstream_branch()
    return success

# class BackupMode(StrEnum):
#     incremental = auto()
#     last_time_only = auto()


class GitHeadHashAcquisitionMode(StrEnum):
    rev_parse = auto()
    log = auto()


from pydantic import Field, BaseModel

# shall you detect if current branch has no upstream branch, and emit exception if so, to prevent 'up-to-date' info being slienced.

# courtesy of ChatGPT

REPO_UP_TO_DATE_KW = "Your branch is up to date with"
REPO_BEHIND_KW = "Your branch is behind"
REPO_AHEAD_OF_KW = "Your branch is ahead of"

NOT_ADDED_KW = "Changes not staged for commit"  # we can have that keyword in many situations, like submodules. we can detect that if we do `git add .` then check status again. if the returned value is not changed, means no need to commit at all.
NOT_COMMITED_KW = "no changes added to commit"
INCOMPLETE_COMMITMENT_KW = "Changes to be committed"
COMMIT_COMPLETE_KW = "nothing to commit, working tree clean"


class RepoStatus(BaseModel):
    up_to_date: bool
    has_unstaged_files: bool
    incomplete_commit: bool

    def need_to_run_commitment_script(self):
        ret = (not self.up_to_date) or self.has_unstaged_files or self.incomplete_commit
        return ret


# you may need to sync description with title to use `pydantic_argparse`.
def get_repo_status():
    out1 = check_repo_status()

    # get the info.
    up_to_date = REPO_UP_TO_DATE_KW in out1
    has_unstaged_files = NOT_ADDED_KW in out1

    if has_unstaged_files:
        ret = os.system(f"{GIT} add .")
        assert ret == 0

        out2 = check_repo_status()
        if out2 == out1:
            has_unstaged_files = False
        incomplete_commit = INCOMPLETE_COMMITMENT_KW in out2
    else:
        incomplete_commit = INCOMPLETE_COMMITMENT_KW in out1

    stat = RepoStatus(
        up_to_date=up_to_date,
        has_unstaged_files=has_unstaged_files,
        incomplete_commit=incomplete_commit,
    )
    return stat


def check_repo_status(encoding="utf-8"):
    proc = subprocess.Popen(
        [GIT, "status"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    b_out, b_err = proc.communicate()

    out = b_out.decode(encoding)
    err = b_err.decode(encoding)
    if err != "":
        raise Exception(
            f"error checking repo status.\n[stdout]\n{out}\n[stderr]\n{err}"
        )
    return out

from typing import Literal

class AtomicCommitConfig(EnvBaseModel):
    # BACKUP_MODE: BackupMode = Field(
    #     default=BackupMode.last_time_only,
    #     title="Backup mode configuration"
    #     # default=BackupMode.last_time_only, description="Backup mode configuration"
    # )
    INSTALL_DIR: str = Field(
        default="",
        title="Directory for installation (if set, after installation the program will exit)",
    )
    # SKIP_CONFLICT_CHECK: bool = Field(
    #     default=False, title="Skip duplication/conflict checks during installation."
    # )
    DUPLICATION_CHECK_MODE: Literal['md5sum', 'filesize'] = Field(
        default='md5sum',
        # default='filesize',
        title="Duplication check mode during installation. Duplicated files will not be copied."
    )
    RCLONE_FLAGS: str = Field(
        default="-P", title="Commandline flags for rclone command"
    )
    BACKUP_UPDATE_CHECK_MODE: BackupUpdateCheckMode = Field(
        default=BackupUpdateCheckMode.commit_and_backup_flag_metadata,
        title="Determines necessarity of backup",
    )
    GIT_HEAD_HASH_ACQUISITION_MODE: GitHeadHashAcquisitionMode = Field(
        default=GitHeadHashAcquisitionMode.rev_parse,
        title="How to acquire git HEAD (latest commit) hash",
    )

    NO_COMMIT: bool = Field(default = False, title = 'Skipping commit action')

    SUBMODULE: bool = Field(default = False, title = 'Skipping recursive installation & excecution due to submodule.')


# from pydantic_argparse import ArgumentParser
# # import humps
# parser = ArgumentParser(
#         model=AtomicCommitConfig,
#         # prog="Example Program",
#         # description="Example Description",
#         # description = " ".join(humps.kebabize(AtomicCommitConfig.__name__).split("-")),
#         version="0.0.1",
#         # epilog="Example Epilog",
#     )
# args = parser.parse_typed_args()

# Print Args
# logger_print(args)
# exit()

config = getConfig(AtomicCommitConfig)

rclone_flags = config.RCLONE_FLAGS
RCLONE_SYNC_CMDGEN = lambda source, target:  f"{RCLONE} sync {rclone_flags} \"{source}\" \"{target}\""

BACKUP_GIT_CONFIG = RCLONE_SYNC_CMDGEN(GIT_CONFIG_ABSPATH, GIT_CONFIG_BACKUP_ABSPATH)
RESTORE_GIT_CONFIG = RCLONE_SYNC_CMDGEN(GIT_CONFIG_BACKUP_ABSPATH, GIT_CONFIG_ABSPATH)

def backup_gitconfig():
    success = False
    success = exec_system_command_and_check_return_code(BACKUP_GIT_CONFIG, 'backup .gitconfig')
    return success

def restore_gitconfig():
    success = False
    success = exec_system_command_and_check_return_code(RESTORE_GIT_CONFIG, 'restore .gitconfig')
    return success


def check_if_filepath_is_valid(filepath):
    assert os.path.exists(filepath), f"path '{filepath}' does not exist."
    assert os.path.isfile(filepath), f"path '{filepath}' is not file."

def checksum(filepath):
    check_if_filepath_is_valid(filepath)
    ret = str(_checksum(filepath))
    return ret

if config.DUPLICATION_CHECK_MODE == 'filesize':
    def _checksum(filepath):
        ret = os.path.getsize(filepath)
        return ret
elif config.DUPLICATION_CHECK_MODE == 'md5sum':
    from simple_file_checksum import get_checksum
    def _checksum(filepath):
        ret = get_checksum(filepath, algorithm='MD5')
        return ret
else:
    raise Exception(f"Unknown DUPLICATION_CHECK_MODE: {config.DUPLICATION_CHECK_MODE}")

# instead of 'where', we have `shutil.which`

# def _access_check(fn, mode):
#     return os.path.exists(fn) and os.access(fn, mode) and not os.path.isdir(fn)


# from typing import Iterator


# def where(cmd, mode=os.F_OK | os.X_OK, path=None) -> Iterator[str]:
#     """Given a command, mode, and a PATH string, generate all paths which
#     conforms to the given mode on the PATH.

#     `mode` defaults to os.F_OK | os.X_OK. `path` defaults to the result
#     of os.environ.get("PATH"), or can be overridden with a custom search
#     path.

#     """
#     # If we're given a path with a directory part, look it up directly rather
#     # than referring to PATH directories. This includes checking relative to the
#     # current directory, e.g. ./script
#     if os.path.dirname(cmd):
#         if _access_check(cmd, mode):
#             return cmd
#         return None

#     use_bytes = isinstance(cmd, bytes)

#     if path is None:
#         path = os.environ.get("PATH", None)
#         if path is None:
#             try:
#                 path = os.confstr("CS_PATH")
#             except (AttributeError, ValueError):
#                 # os.confstr() or CS_PATH is not available
#                 path = os.defpath
#         # bpo-35755: Don't use os.defpath if the PATH environment variable is
#         # set to an empty string

#     # PATH='' doesn't match, whereas PATH=':' looks in the current directory
#     if not path:
#         return None

#     if use_bytes:
#         path = os.fsencode(path)
#         path = path.split(os.fsencode(os.pathsep))
#     else:
#         path = os.fsdecode(path)
#         path = path.split(os.pathsep)

#     if sys.platform == "win32":
#         # The current directory takes precedence on Windows.
#         curdir = os.curdir
#         if use_bytes:
#             curdir = os.fsencode(curdir)
#         if curdir not in path:
#             path.insert(0, curdir)

#         # PATHEXT is necessary to check on Windows.
#         pathext_source = os.getenv("PATHEXT") or _WIN_DEFAULT_PATHEXT
#         pathext = [ext for ext in pathext_source.split(os.pathsep) if ext]

#         if use_bytes:
#             pathext = [os.fsencode(ext) for ext in pathext]
#         # See if the given file matches any of the expected path extensions.
#         # This will allow us to short circuit when given "python.exe".
#         # If it does match, only test that one, otherwise we have to try
#         # others.
#         if any(cmd.lower().endswith(ext.lower()) for ext in pathext):
#             files = [cmd]
#         else:
#             files = [cmd + ext for ext in pathext]
#     else:
#         # On other platforms you don't have things like PATHEXT to tell you
#         # what file suffixes are executable, so just pass on cmd as-is.
#         files = [cmd]

#     seen = set()
#     for dir in path:
#         normdir = os.path.normcase(dir)
#         if not normdir in seen:
#             seen.add(normdir)
#             for thefile in files:
#                 name = os.path.join(dir, thefile)
#                 if _access_check(name, mode):
#                     yield name


# for path in where('git'): # multiple git now.
#   logger_print(path)

# use rclone instead?
# REQUIRED_BINARIES = ["bash", "timemachine", "rsync", "git"]
# if on windows, we check if bash is not coming from wsl.
# WSL_BASH = ...

for reqbin in REQUIRED_BINARIES:
    assert (
        shutil.which(reqbin) is not None
    ), f"Required binary '{reqbin}' is not in PATH."

# import pytz
# import datetime

# TIMEZONE = ...

FSCK = f"{GIT} fsck"
LOG_HASH = f'{GIT} log -1 --format=format:"%H"'
REV_PARSE_HASH = f"{GIT} rev-parse HEAD"

GITIGNORE = ".gitignore"
GITIGNORE_INPROGRESS = ".inprogress_gitignore"
GITIGNORE_BACKUP = f"{GITIGNORE}_backup"
ROLLBACK_INPROGRESS_FLAG = ".rollback_inprogress"

GITDIR = ".git"
COMMIT_FLAG = ".atomic_commit_flag"
LOCKFILE = f".atomic_commit_lock{'_nt' if os.name == 'nt' else ''}"
LOCK_TIMEOUT = 5
BACKUP_BASE_DIR = ".git_backup"
BACKUP_GIT_DIR = os.path.join(BACKUP_BASE_DIR, GITDIR)
BACKUP_FLAG = os.path.join(BACKUP_BASE_DIR, ".atomic_backup_flag")
INPROGRESS_DIR = os.path.join(BACKUP_BASE_DIR, ".inprogress")
# INPROGRESS_INCREMENTAL_DIR = os.path.join(BACKUP_BASE_DIR, ".inprogress_incremental")
# INCREMENTAL_BACKUP_DIR_FORMAT = ...
# INCREMENTAL_BACKUP_DIR_GENERATOR = lambda: ...
LOG_DIR = "logs"
IGNORED_PATHS = [LOG_DIR, BACKUP_BASE_DIR, COMMIT_FLAG, LOCKFILE, GITIGNORE_INPROGRESS]
GIT_RM_CACHED_CMDGEN = lambda p: f"{GIT} rm -r --cached {p}"


def git_fsck():
    exit_code = os.system(FSCK)
    success = exit_code == 0
    logger_print(
        f"{GIT} fsck {'success' if success else 'failed'} at: {os.path.abspath('.')}"
    )
    return success


from contextlib import contextmanager


@contextmanager
def chdir_context(dirpath: str):
    cwd = os.getcwd()
    os.chdir(dirpath)
    try:
        yield
    finally:
        os.chdir(cwd)

def detect_upstream_branch_add_safe_directory_and_git_fsck():
    success = False
    detect_upstream_branch_and_add_safe_directory()
    assert os.path.isdir(GITDIR), "Git directory not found!"
    success = git_fsck()
    return success

# BACKUP_COMMAND_COMMON = f"{RCLONE} sync {rclone_flags} {GITDIR} {INPROGRESS_DIR}"
BACKUP_COMMAND_COMMON = RCLONE_SYNC_CMDGEN(GITDIR, INPROGRESS_DIR)

# ROLLBACK_COMMAND = f"{RCLONE} sync {rclone_flags} {BACKUP_GIT_DIR} {GITDIR}"
ROLLBACK_COMMAND = RCLONE_SYNC_CMDGEN(BACKUP_GIT_DIR, GITDIR)


def rollback():
    # do we have incomplete backup? if so, we cannot rollback.
    success = False
    incomplete = os.path.exists(INPROGRESS_DIR)
    if incomplete:
        raise Exception("Backup is incomplete. Cannot rollback.")
    else:
        pathlib.Path(ROLLBACK_INPROGRESS_FLAG).touch()
        return_code = os.system(ROLLBACK_COMMAND)
        assert (
            return_code == 0
        ), f"Running rollback command failed with exit code {return_code}"
        # if config.BACKUP_MODE == BackupMode.incremental:
        # ...  # group files based on modification time, or `--min-age`
        # # selected files in main dir along with files from backup dir
        git_not_corrupted = git_fsck()
        success = git_not_corrupted
        os.remove(ROLLBACK_INPROGRESS_FLAG)
    return success


def install_script(install_dir:str, source_dir:str = "."):
    # if config.INSTALL_DIR is not "":
    assert install_dir != source_dir, f"install_dir '{install_dir}' shall not be the same as source_dir '{source_dir}'"
    success = False
    if os.path.exists(install_dir):
        with chdir_context(install_dir):
            # add_safe_directory()
            success = False
            try: # restore first.
                success = detect_upstream_branch_add_safe_directory_and_git_fsck()
                assert success, "First installation failed."
            except:
                print("Trying second installation")
                rollback_success = rollback()
                if rollback_success:
                    success = detect_upstream_branch_add_safe_directory_and_git_fsck()
            if not success:
                raise Exception("Target git repository is corrupted.")

        localfiles = os.listdir(source_dir)
        install_files = ['atomic_commit.py', 'config_utils.py', 'exception_utils.py', 'log_utils.py', 'argparse_utils.py', 'exceptional_print.py', 'error_utils.py']
        for f in install_files:
            assert f in localfiles, "Could not find '%s' in '%s'" % (f, source_dir)
        # install_files = [f for f in localfiles if f.endswith(".py")] # let's redefine this.
        # print(install_files)
        # breakpoint()
        # if not config.SKIP_CONFLICT_CHECK:
        target_dir_files = os.listdir(install_dir)
        need_install_files = []
        for f in install_files:
            if f not in target_dir_files:
                logger_print("target directory %s does not exist" % f)
                need_install_files.append(f)
            else:
                target_checksum = checksum(os.path.join(install_dir, f))
                install_checksum = checksum(os.path.join(source_dir, f))
                if target_checksum == install_checksum:
                    logger_print(f"skipping installation of file '{f}' due to same checksum '{install_checksum}' (method: {config.DUPLICATION_CHECK_MODE})")
                else:
                    logger_print(f"file '{f}' checksum mismatch. (target: '{target_checksum}', install: '{install_checksum}')")
                    need_install_files.append(f)
        # conflict_files = [f for f in install_files if f in target_dir_files]
        # if set(conflict_files) == set(install_files):
        #     raise Exception(
        #         "You probably have installed at directory %s" % config.INSTALL_DIR
        #     )
        # if conflict_files != []:
        #     err = [
        #         f"Conflict file '{f}' found in target directory '{config.INSTALL_DIR}'"
        #         for f in conflict_files
        #     ]
        #     raise Exception("\n".join(err))
        
        for f in need_install_files:
            logger_print(f"installing '{f}' to '{install_dir}'")
            target_fpath = os.path.join(install_dir, f)
            source_fpath = os.path.join(source_dir,f)
            shutil.copy(source_fpath, target_fpath)
        logger_print(f"Atomic commit script installed at: '{install_dir}'")
        success = True
    else:
        raise Exception(
            f"Could not find installation directory at '{install_dir}'"
        )
    return success

if config.INSTALL_DIR != "":
    success = install_script(config.INSTALL_DIR)
    if success:
        exit(0)
    else:
        raise Exception(f"Installation failed at '{config.INSTALL_DIR}' for unknown reason.")

assert os.path.isdir(GITDIR), "Git directory not found!"
if os.path.exists(BACKUP_BASE_DIR):
    if not os.path.isdir(BACKUP_BASE_DIR):
        raise Exception(
            f"Backup base directory path '{BACKUP_BASE_DIR}' is not a directory."
        )
else:
    os.mkdir(BACKUP_BASE_DIR)

gitignore_content = ""
existing_ignored_paths = []

if os.path.exists(GITIGNORE_BACKUP):
    if os.path.exists(GITIGNORE):
        raise Exception(
            f"Both '{GITIGNORE}' and '{GITIGNORE_BACKUP}' exist. Cannot recover."
        )
    else:
        print(f"Restoring '{GITIGNORE}' from backup")
        shutil.move(GITIGNORE_BACKUP, GITIGNORE)

if os.path.exists(GITIGNORE):
    if os.path.isfile(GITIGNORE):
        with open(GITIGNORE, "r") as f:
            gitignore_content = f.read()
            existing_ignored_paths = gitignore_content.split("\n")
            existing_ignored_paths = [t.strip() for t in existing_ignored_paths]
            existing_ignored_paths = [t for t in existing_ignored_paths if len(t) > 0]

line = lambda s: f"{s.strip()}\n"
missing_ignored_paths = []

for p in IGNORED_PATHS:
    if p not in existing_ignored_paths:
        missing_ignored_paths.append(p)
        cmd = GIT_RM_CACHED_CMDGEN(p)
        ret = os.system(cmd)
        if ret not in [0, 128]:
            logger_print(
                f"warning: abnormal exit code {ret} while removing path '{p}' from git cache"
            )

has_gitignore = False
if missing_ignored_paths != []:
    with open(GITIGNORE_INPROGRESS, "w+") as f:
        if gitignore_content != "":
            f.write(line(gitignore_content))
        for p in missing_ignored_paths:
            # for p in existing_ignored_paths + missing_ignored_paths:
            f.write(line(p))
    if os.path.exists(GITIGNORE):
        has_gitignore = True
        shutil.move(GITIGNORE, GITIGNORE_BACKUP)
    shutil.move(GITIGNORE_INPROGRESS, GITIGNORE)
    if has_gitignore:
        os.remove(GITIGNORE_BACKUP)




def get_git_head_hash():
    if config.GIT_HEAD_HASH_ACQUISITION_MODE == GitHeadHashAcquisitionMode.log:
        cmd = LOG_HASH
    else:
        cmd = REV_PARSE_HASH

    proc = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
    assert (
        code := proc.returncode
    ) == 0, f"Checking lastest commit hash failed with exit code {code}."
    _hash = proc.stdout.strip().decode("utf8")
    return _hash


# you may find usable bash shell next to our git executable on windows, and it is preferred
# because wsl bash sucks


def get_script_path_and_exec_cmd(script_prefix):
    """
    Get the script path and the command to execute the script.

    Args:
        script_prefix (str): The prefix of the script name.

    Returns:
        tuple: A tuple containing the script path (str) and the command to execute the script (str).
    """
    script_path = f"{script_prefix}.py"
    exec_prefix = sys.executable
    if not os.path.exists(script_path):
        if os.name == "nt":
            script_suffix = "cmd"
            exec_prefix = "cmd /C"
        else:
            script_suffix = "sh"
            exec_prefix = "bash"
        script_path = f"{script_prefix}.{script_suffix}"
        assert os.path.exists(
            script_path
        ), f"failed to find os native implementation of commit script: {script_path}"
    else:
        logger_print(f"using os independent implementation of commit: '{script_path}'")

    cmd = f"{exec_prefix} {script_path}"
    return script_path, cmd


# deadlock: if both backup integrity & fsck failed, what to do?
# when backup is done, put head hash as marker
# default skip check: mod-time & size


# if config.BACKUP_MODE == BackupMode.last_time_only:
# BACKUP_COMMAND_GEN = lambda: BACKUP_COMMAND_COMMON
# else:
#     # take care of last backup!
#     BACKUP_COMMAND_GEN = (
#         lambda: f"{BACKUP_COMMAND_COMMON} '--backup-dir={INPROGRESS_INCREMENTAL_BACKUP_DIR}'"
#     )


def backup():
    if os.path.exists(BACKUP_GIT_DIR):
        shutil.move(BACKUP_GIT_DIR, INPROGRESS_DIR)
    backup_command = BACKUP_COMMAND_COMMON
    # backup_command = BACKUP_COMMAND_GEN()
    success = exec_system_command_and_check_return_code(backup_command, 'backup')
    # then we move folders into places.
    shutil.move(INPROGRESS_DIR, BACKUP_GIT_DIR)
    # if config.BACKUP_MODE == BackupMode.incremental:
    #     incremental_backup_dir = INCREMENTAL_BACKUP_DIR_GENERATOR()
    #     shutil.move(INPROGRESS_INCREMENTAL_BACKUP_DIR, incremental_backup_dir)
    # create backup flag.
    if (
        config.BACKUP_UPDATE_CHECK_MODE
        == BackupUpdateCheckMode.commit_and_backup_flag_metadata
    ):
        pathlib.Path(BACKUP_FLAG).touch()
        # use os.path.getmtime(BACKUP_FLAG) to get the latest timestamp
    else:  # write git hash to flag.
        with open(BACKUP_FLAG, "w+") as f:
            git_hash = get_git_head_hash()
            f.write(git_hash)
    return success


def get_last_backup_commit_hash():
    _hash = None
    if os.path.isfile(BACKUP_FLAG):
        with open(BACKUP_FLAG, "r") as f:
            _hash = f.read().strip()
    return _hash


INF = 1e20
from typing import Optional


def get_file_mtime_with_default(fpath: str, default: Optional[float] = None):
    if os.path.exists(fpath):
        if os.path.isfile(fpath):
            return os.path.getmtime(fpath)
        else:
            raise Exception(
                "Cannot get mtime of file '%s' because non-file object is taking place of it."
                % fpath
            )
    else:
        if default is not None:
            return default
        else:
            raise Exception(
                "Cannot get mtime of file '%s' because it does not exist and default mtime is not set."
                % fpath
            )


def atomic_backup():
    """
    atomic backup:

    assumed passed fsck

    1. inprogress check: if has .inprogress folder, just continue backup
    2. backup integrity check: check if marker equals to git head/status. if not, then backup.
    """
    need_backup = False
    success = False

    if os.path.exists(INPROGRESS_DIR):
        need_backup = True
    else:
        if config.BACKUP_UPDATE_CHECK_MODE == BackupUpdateCheckMode.git_commit_hash:
            last_backup_commit_hash = get_last_backup_commit_hash()
            current_commit_hash = get_git_head_hash()
            if current_commit_hash != last_backup_commit_hash:
                need_backup = True
        else:
            # compare mtime.
            mtime_backup_flag = get_file_mtime_with_default(BACKUP_FLAG, default=-INF)
            mtime_commit_flag = get_file_mtime_with_default(COMMIT_FLAG, default=INF)
            if mtime_commit_flag > mtime_backup_flag:
                need_backup = True

    if need_backup:
        success = backup()
    else:
        success = True
    return success


#####################################
# RCLONE BACKUP RESTORATION DIAGRAM #
#####################################
#
# time
#  |   current   | back_0 | back_1 | back_2 |  state   |
#  |-------------|--------|--------|--------|----------|
# 0| a b c       |  *     |  *     |        | *a *b *c |
# 1|       +d    |        |        |     *  | a b c *d |
# 2|     -c      |  c     |        |        |  a b d   |
# 3|   +b+c      |        |  b     |        | a *b d *c|
# 4|       -d    |        |        |     d  |  a b    c|
#
# unless you make modification records with timestamp per state, you cannot restore every state.
# you must do more than just using rclone flags.
# given its complexity, let's not do it.
#


COMMIT_CMD = ...
if not config.NO_COMMIT:
    _, COMMIT_CMD = get_script_path_and_exec_cmd("commit")


def commit():
    logger_print(f"running commit command: {COMMIT_CMD}")
    success = False
    # add_config_success = add_safe_directory()
    # if add_config_success:
    return_code = os.system(COMMIT_CMD)
    assert (
        return_code == 0
    ), f"Failed to execute commit script with exit code {return_code}"
    success = True
    return success


# TODO: formulate this into a state machine.
from easyprocess import EasyProcess
def execute_script_submodule(directory:str):
    success = False
    cmd = [sys.executable, SCRIPT_FILENAME]
    cmd.extend(['--no_commit', 'True', '--submodule', 'True'])
    with chdir_context(directory):
        proc = EasyProcess(cmd).call()
        ret = proc.return_code
        success = ret == 0
        if not success:
            logger_print(f"Failed to execute script at directory '{directory}'")
    return success


def recursive_install_and_execute_script_to_lower_git_directories():
    success = False
    candidate_dirs = set()
    
    # TODO: skip symlinks
    for dirpath, dirnames, filenames in os.walk(".", followlinks = False):
        if BACKUP_BASE_DIR not in dirpath and os.path.basename(dirpath) == GITDIR:
            submodule_dir, _ = os.path.split(dirpath)
            if submodule_dir != '.':
                logger_print(f"adding submodule git directory '{submodule_dir}'")
                candidate_dirs.add(submodule_dir)
    if len(candidate_dirs) == 0:
        logger_print(f"no submodule git directory found")
        success = True
    for install_dir in candidate_dirs:
        logger_print(f"installing and executing script at submodule '{install_dir}")
        success = install_script(install_dir, source_dir = ".")
        if success:
            success = execute_script_submodule(install_dir) # enable SUBMODULE & NO_COMMIT
        if not success:
            raise Exception(f"submodule execution failed at: '{install_dir}'")
    return success


# TODO: check necessarity of commitment
# TODO: check if commit is incomplete
def atomic_commit():
    r"""
      fsck 
    /      \
  succ     fail
      \     | rollback (most recent backup)
  d_comm > d_back? # can be replaced by metadata check or `git rev-parse HEAD` (equivalent to: `git log -1 --format=format:"%H"` alongside metadata check with depth=1 (if possible to retrieve from timemachine current backup)
      | y     | n
    backup (atomic) & update d_back nothing
      \       /
      need_commit?
        | y     \ n
       commit    exit
          |
        fsck
    succ/  \fail
        |   \_ rollback (most recent backup) & exit
    incomplete? -(y)- exit
        | n
    update d_comm
      | backup (atomic)
      | update d_back
      | exit
  """
    success = False
    commit_success = False
    commit_hash_changed = False
    # add_safe_directory()
    detect_upstream_branch_and_add_safe_directory()
    # add_config_success = add_safe_directory()
    # if not add_config_success:
    #     logger_print("failed to add safe directory")
    #     return success

    # now we recursively install and execute this script (skipping commit) to lower `.git` directories, skipping symbolic links
    if not config.SUBMODULE:
        recursive_install_and_execute_script_to_lower_git_directories()

    can_commit = atomic_commit_common()

    if config.NO_COMMIT:
        logger_print("skipping commit action because configuration")
        return can_commit

    if can_commit:
        status = get_repo_status()
        if status.need_to_run_commitment_script():
            hash_before = get_git_head_hash()
            commit_success = commit()
            hash_after = get_git_head_hash()
            commit_hash_changed = hash_after != hash_before
            if not commit_success:
                return success
        else:
            logger_print(
                "Repo status:", status, "No need to run commit script", "Exiting"
            )
            success = True
            return success

    finalize_commit_success = atomic_commit_common(
        post_commit=True,
        commit_success=commit_success,
        commit_hash_changed=commit_hash_changed,
    )
    if finalize_commit_success:
        success = True

    return success


def atomic_commit_common(
    post_commit=False, commit_success=..., commit_hash_changed=...
):
    git_not_corrupted = False
    can_commit = False
    # rollback_inprogress = os.path.exists(ROLLBACK_INPROGRESS_FLAG)
    git_not_corrupted = git_fsck()

    if post_commit:
        post_commit_actions(commit_success, commit_hash_changed)

    if git_not_corrupted:
        if atomic_backup():
            can_commit = True
    elif rollback():
        can_commit = True
    else:
        logger_print("Rollback failed, exiting")
    return can_commit


def post_commit_actions(commit_success, commit_hash_changed):
    if commit_success:
        if commit_hash_changed:
            pathlib.Path(COMMIT_FLAG).touch()
        else:
            status = get_repo_status()
            logger_print("Repo status:", status)
            if status.need_to_run_commitment_script():
                logger_print("Incomplete commitment detected.")
            raise Exception("Commitment failed.")


if __name__ == "__main__":
    with filelock.FileLock(LOCKFILE, timeout=LOCK_TIMEOUT) as lockfile:
        success = atomic_commit()
        if not success:
            raise Exception("Failed to perform atomic commit.")
        else:
            logger_print("Cleaning up after successful commit:")
            os.system(PRUNE_NOW)

