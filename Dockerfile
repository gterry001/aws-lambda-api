FROM public.ecr.aws/lambda/python:3.9

# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código
COPY app ./app

# Definir el handler
CMD ["app.main.handler"]
