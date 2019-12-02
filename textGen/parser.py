import argparse
import os
from tqdm import trange


def read_all(root, ham_only=False):
    if not os.path.exists(root):
        raise ValueError("Path Doesn't Exist")

    out_path = root + "/out"
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if not os.path.exists(out_path + "/ham"):
        os.mkdir(out_path +"/ham")
    if not os.path.exists(out_path + "/spam"):
        os.mkdir(out_path +"/spam")

    spam_c = 0
    ham_c = 0

    err_c = 0

    with open(root + "/full/index", 'r') as index:

        counter = 0 # only count printed files, ie don't count skipped in case of ham only
        for c in trange(rawgencount(root + "/full/index")):
            iline = index.readline()
            (label, msg_path) = iline.split(" ..", 1)
            if label == 'spam' and ham_only:
                continue

            try:
                with open(root + msg_path.strip(), 'r') as msg:
                    try:
                        _, body = msg.read().split('\n\n', 1) # strip the header info
                    except ValueError:
                        err_c += 1
                        continue
            except FileNotFoundError:
                err_c += 1
                continue
            except OSError:
                err_c += 1
                continue
                
            
            with open(out_path + f"/{label}/{counter:06d}", 'w+') as fout:
                fout.write(body)
                counter += 1

            if label == "spam":
                spam_c += 1
            else:
                ham_c += 1

    print(f"Ham: {ham_c}\nSpam: {spam_c}\nErr: {err_c}")

        


# https://stackoverflow.com/questions/845058/how-to-get-line-count-of-a-large-file-cheaply-in-python?page=1&tab=votes#tab-top
def _make_gen(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024*1024)

# https://stackoverflow.com/questions/845058/how-to-get-line-count-of-a-large-file-cheaply-in-python?page=1&tab=votes#tab-top
def rawgencount(filename):
    f = open(filename, 'rb')
    f_gen = _make_gen(f.raw.read)
    return sum( buf.count(b'\n') for buf in f_gen )

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, required=True)
    parser.add_argument('--ham_only', action='store_true')

    args = parser.parse_args()

    read_all(args.path, args.ham_only)


if __name__ == "__main__":
    main()