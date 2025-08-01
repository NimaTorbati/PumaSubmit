#!/usr/bin/env bash
apt-get update && apt-get install -y pv
# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Set default container name
container_tag="puma-challenge-baseline-track1"

# Check if an argument is provided
if [ "$#" -eq 1 ]; then
    container_tag="$1"
fi

# Get the build information from the Docker image tag
build_timestamp=$( docker inspect --format='{{ .Created }}' "$container_tag")

if [ -z "$build_timestamp" ]; then
    echo "Error: Failed to retrieve build information for container $container_tag"
    exit 1
fi

# Format the build information to remove special characters
formatted_build_info=$(date -d "$build_timestamp" +"%Y%m%d_%H%M%S")

# Set the output filename with timestamp and build information
output_filename="${SCRIPT_DIR}/${container_tag}_${formatted_build_info}.tar.gz"

# Get the estimated size of the Docker image
image_size=$(docker inspect --format='{{ .Size }}' "$container_tag")

# Save the Docker container and gzip it
docker save "$container_tag" | pv -s "$image_size" | gzip -c > "$output_filename"

echo "Container saved as ${output_filename}"
