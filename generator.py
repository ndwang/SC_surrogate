import argparse
from distgen import Generator
from distgen.writers import writer

def generate_beam(template, output_file):
    gen = Generator(template)
    beam = gen.beam()
    writer("openPMD", beam, output_file)

def main():
    parser = argparse.ArgumentParser(description="Generate a beam using Distgen.")
    parser.add_argument("template", help="Path to the Distgen YAML configuration file")
    parser.add_argument("output_file", help="Path to the output file")
    args = parser.parse_args()

    generate_beam(args.template, args.output_file)

if __name__ == "__main__":
    main()


