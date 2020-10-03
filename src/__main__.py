import sys
from enum import Enum

import src.bin.xor.Main as xor

def switch(project):
    Projects = {
        # add the main func of projects to this dictionary to run them with command line arguments
        "xor": xor.run
    }
    return Projects.get(project, "No such project")

if __name__ == '__main__':
    if switch(sys.argv[1]) == "No such project":
        print("No such project")
    else:
        switch(sys.argv[1])()