FROM python:3.12-slim-bookworm

WORKDIR /app

# Install base packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg2 \
        unixodbc \
        unixodbc-dev && \
    rm -rf /var/lib/apt/lists/*

# Add Microsoft GPG key and repository
RUN mkdir -p /usr/share/keyrings && \
    curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
    | gpg --dearmor -o /usr/share/keyrings/microsoft.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/microsoft.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" \
    > /etc/apt/sources.list.d/mssql-release.list

# Install the MS ODBC Driver 18
RUN apt-get update && \
    ACCEPT_EULA=Y apt-get install -y --no-install-recommends \
    msodbcsql18 && \
    rm -rf /var/lib/apt/lists/*

# Install Docker CLI
RUN apt-get update && \
    apt-get install -y --no-install-recommends lsb-release && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg \
      | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
      https://download.docker.com/linux/debian \
      $(lsb_release -cs) stable" \
      > /etc/apt/sources.list.d/docker.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends docker-ce-cli && \
    rm -rf /var/lib/apt/lists/*

COPY code/requirements.txt code/constraints.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

# Copy the code directory
COPY code/ ./code/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default entrypoint 
ENTRYPOINT ["python"]