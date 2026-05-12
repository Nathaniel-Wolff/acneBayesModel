#using architecture with Python model fitting and result calculation
#R Shiny sends UI results to restAPI, used in calculations, and sent back to R frontend

FROM python:3.11-slim
WORKDIR /DockerAcneBayesModel

#making sure build tools are installed
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

#Dependencies with modularized code
COPY pyproject.toml . 
COPY src/ ./src/

RUN pip install --no-cache-dir .


#copying model config for now, later will be allowed to be changed by user
RUN mkdir -p ./data
COPY bhm_model_config.json ./data/
COPY local_model_run/sim_acne_amended_v2.csv ./data/
COPY local_model_run/sim_acne_diet.csv ./data/ 

#ENV PYTHONPATH="/DockerAcneBayesModel/src:${PYTHONPATH}"

ENTRYPOINT ["python", "-m", "acne_model.model_script"]

