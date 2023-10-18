FROM python:3.7-stretch

ARG BUILD_ENV
ARG OVERCOOKED_BRANCH
ARG HARL_BRANCH
ARG GRAPHICS

WORKDIR /app
RUN echo "deb http://archive.debian.org/debian stretch main" > /etc/apt/sources.list

RUN apt-get -y update
RUN apt-get install -y vim
RUN apt-get install -y nano

# Install non-chai dependencies
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Install eventlet production server if production build
RUN if [ "$BUILD_ENV" = "production" ] ; then pip install eventlet ; fi

# Clone chai code
RUN git clone --recursive https://github.com/JoaninhaRaposo/overcooked_ai.git --branch $OVERCOOKED_BRANCH --single-branch /overcooked_ai

# Dummy data_dir so things don't break
RUN echo "import os; DATA_DIR=os.path.abspath('.')" >> /overcooked_ai/src/human_aware_rl/data_dir.py

# Install chai dependencies
RUN pip install -e '/overcooked_ai[harl]'


RUN apt-get install -y libgl1-mesa-dev



# Copy over remaining files
COPY ./static ./static
COPY ./*.py ./
COPY ./graphics/$GRAPHICS ./static/js/graphics.js
COPY ./config.json ./config.json



# Set environment variables that will be used by app.py

#Use the following host for docker server
ENV HOST 146.193.226.27 

#Use the following HOST for the DLAI server
#ENV HOST 0.0.0.0 
ENV PORT 5000
ENV CONF_PATH config.json

# Do the thing
EXPOSE 5000
CMD ["python", "-u", "app.py"]
