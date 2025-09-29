FROM public.ecr.aws/lambda/python:3.9


# Instalar dependencias del sistema necesarias para compilar pycares (y otras libs)
RUN yum install -y gcc gcc-c++ make python3-devel c-ares-devel
# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --target /var/task

# Copiar el código
COPY app ./app

# Definir el handler
CMD ["app.main.handler"]


