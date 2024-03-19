import docker
import os

docker_client = docker.from_env()
docker_tags = {tag.split(":")[0] for image in docker_client.images.list() for tag in image.tags}
for tag in [tag for tag in docker_tags if tag.startswith("ann-benchmarks-")]:
	os.system(f'docker save {tag} -o {tag}.tar')
	os.system(f'singularity build {tag}.sif docker-archive://{tag}.tar')
	os.system(f'rm {tag}.tar')