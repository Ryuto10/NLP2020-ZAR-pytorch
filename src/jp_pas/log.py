import os
import subprocess
import sys
from datetime import datetime
import json


class StandardLogger(object):
    def __init__(self, file):
        self.file = os.path.abspath(file)
        self.n_file_overlap = 1
        self.name_file_overlap = self.file

    def write(self, args=None, comment=None):
        """
        Make text logs
        """
        # Check overlap of the file name
        while os.path.exists(self.name_file_overlap):
            self.n_file_overlap += 1
            self.name_file_overlap = self.file.replace(".", "_" + str(self.n_file_overlap) + ".", 1)
        self.file = self.name_file_overlap

        print("# Create log file: {}".format(self.file))
        with open(self.file, "w") as fo:
            fo.write("# standard\n")

        self._write_standard_log()

        if os.path.exists('.git'):
            self._write_git_log()

        if args:
            self._write_args_log(args)

        if comment:
            self._write_comment_log(comment)

    def write_endtime(self):
        with open(self.file, "a") as fo:
            fo.write("\n\n# end time\n")
            fo.write(get_time())

    def _write_standard_log(self):
        with open(self.file, "a") as fo:
            fo.write("## command\n")
            fo.write(" ".join(sys.argv))
            fo.write("\n\n## directory\n" + os.getcwd() + "\n\n## start time\n")
            fo.write(get_time())

    def _write_git_log(self):
        with open(self.file, "a") as fo:
            gitdiff = subprocess.check_output("git diff HEAD " + sys.argv[0], shell=True).decode('utf-8')
            gitlog = subprocess.check_output("git log --pretty=fuller | head -7", shell=True).decode('utf-8')

            fo.write("  \n\n# Git\n## log\n> ")
            fo.write("  \n> ".join(_ for _ in gitlog.split("\n") if _.strip()))
            fo.write("  \n## diff\n> \ ")
            fo.write('  \n> \ '.join(_ for _ in gitdiff.split("\n")))
            fo.write('  \n')

    def _write_args_log(self, args):
        args_dict = args.__dict__
        for k, v in args_dict.items():
            if type(v) == str and os.path.exists(v):
                args_dict[k] = os.path.abspath(v)

        with open(self.file, "a") as fo:
            fo.write("\n\n# argparse\n")
            fo.write('  \n'.join((k + " : " + str(v)) for k, v in args_dict.items()))

    def _write_comment_log(self, comment):
        with open(self.file, "a") as fo:
            fo.write("\n\n# comment\n")
            fo.write(comment)

    def add(self, sentence):
        with open(self.file, "a") as fo:
            fo.write(sentence)


def get_time():
    return datetime.today().strftime("%Y/%m/%d %H:%M:%S")


def write_args_log(args, file):
    if os.path.exists(file):
        raise FileExistsError("'{}' already exists.".format(file))

    args_dict = args.__dict__
    for k, v in args_dict.items():
        if type(v) == str and os.path.exists(v):
            args_dict[k] = os.path.abspath(v)

    with open(file, "w") as fo:
        json.dump(args_dict, fo)
