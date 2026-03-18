# Заполнить файл .env


# установка и запуск
## 1 вариант
1. Docker-образ например с именем bot_advanced:v1:
docker build -t bot_advanced:v1 .

2. запуск контейнера
docker run --rm --name bot-container-new --env-file .env -v ${PWD}:/app bot_advanced:v1


## 2 вариант
docker-compose up --build


# остановка работы контейнера 
docker stop <container_id>


