FROM snakemake/snakemake:latest

LABEL author="Mario Senden"
LABEL email="mario.senden@maastrichtuniversity.nl"

# Set the working directory
WORKDIR /workflow

# copy everying from the current directory to the working directory
COPY . /workflow
RUN pip install -r requirements.txt

# Run the Snakemake workflow when the container starts
CMD ["snakemake", "--cores", "30"]