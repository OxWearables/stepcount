import argparse
import os



def generate_commands(input_dir, output_dir, cmdsfile='list-of-commands.txt', fext="cwa", cmdopts=""):
    """Generate a text file listing processing commands for files found under input_dir/

    :param str input_dir: Directory containing accelerometer files to process.
    :param str output_dir: Name for output directory to store the processing results.
    :param str cmdsfile: Name for generated file with list of commands.
    :param str fext: Accelerometer file extension e.g. cwa, CWA, bin, BIN, gt3x...
    :param str cmdopts: String of options to pass e.g. "--type rf" to use Random Forest classifier.

    :return: New file written to <cmdsfile>
    :rtype: void

    :Example:
    >>> import utils
    >>> utils.generate_commands("MyAccFiles/", output_dir="MyOutputs/", cmdsfile="list-of-commands.txt", fext="cwa", cmdopts="--type rf")
    <list of commands written to "list-of-commands.txt">
    """

    # List all accelerometer files under input_dir/
    fpaths = []
    fext = fext.lower()
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith((
                fext,
                fext + ".gz",
                fext + ".zip",
                fext + ".bz2",
                fext + ".xz"
            )):
                fpaths.append(os.path.join(root, file))
    print(f"Found {len(fpaths)} accelerometer files...")

    with open(cmdsfile, 'w') as f:
        for fpath in fpaths:

            # Use the file name as the output folder name for the process,
            # keeping the same directory structure of input_dir/
            # Example: If fpath is {input_dir}/group0/subject123.cwa then
            # _output_dir will be {output_dir}/group0/subject123/
            _output_dir = fpath.replace(input_dir.rstrip("/"), output_dir.rstrip("/")).split(".")[0]

            cmd = f"stepcount '{fpath}' --outdir '{_output_dir}' {cmdopts}"

            f.write(cmd)
            f.write('\n')

    print('List of commands written to ', cmdsfile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('--output_dir', '-d', required=True)
    parser.add_argument('--cmdsfile', '-f', type=str, default='list-of-commands.txt')
    parser.add_argument('--fext', '-a', default='cwa', help='Acc file type e.g. cwa, CWA, bin, BIN, gt3x...')
    parser.add_argument('--cmdopts', '-x', type=str, default="", help="String of processing options e.g. '--type rf' to use Random Forest classifier.")
    args = parser.parse_args()

    generate_commands(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        cmdsfile=args.cmdsfile,
        fext=args.fext,
        cmdopts=args.cmdopts
    )


if __name__ == '__main__':
    main()
