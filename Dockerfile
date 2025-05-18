FROM nvidia/cuda:12.4.1-base-ubuntu22.04

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        git && \
    rm -rf /var/lib/apt/lists/*


# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Copy the project into the image
ADD . /app

# Sync the project into a new environment, asserting the lockfile is up to date
RUN uv sync --locked