FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY data/raw ./data/raw
RUN pip install --no-cache-dir .
ENTRYPOINT ["tpta"]
CMD ["instruments"]

