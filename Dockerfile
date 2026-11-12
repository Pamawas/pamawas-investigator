# syntax=docker/dockerfile:1
FROM python:3.14-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --disable-pip-version-check -r requirements.txt

COPY . .

ARG VERSION=""
RUN if [ -n "$VERSION" ]; then printf '%s\n' "$VERSION" > /app/version.txt; elif [ ! -f /app/version.txt ]; then printf 'dev-unknown\n' > /app/version.txt; fi

FROM python:3.14-slim

ARG VERSION=""
ENV APP_VERSION=${VERSION}

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.14/site-packages /usr/local/lib/python3.14/site-packages
COPY --from=builder /app /app
COPY --from=builder /app/version.txt ./version.txt

EXPOSE 8080

CMD ["python", "main.py"]