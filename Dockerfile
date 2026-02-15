diff --git a/dockerfile b/dockerfile
index 2043aa98704924d52fc5fe3deef0e881c5a5b481..90410b65c1db43d523ba229c74b55c56802bcf9c 100644
--- a/dockerfile
+++ b/dockerfile
@@ -1,12 +1,13 @@
-FROM python:3.12-slim
-
-WORKDIR /app
-COPY requirements.txt .
-RUN pip install --no-cache-dir -r requirements.txt
-
-COPY main.py .
-
-ENV PORT=8080
-EXPOSE 8080
-
-CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
+FROM python:3.12-slim
+
+WORKDIR /app
+COPY requirements.txt .
+RUN pip install --no-cache-dir -r requirements.txt
+
+COPY main.py .
+COPY index.html .
+
+ENV PORT=8080
+EXPOSE 8080
+
+CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
