
.PHONY: build
build:
	sudo docker build -t pyhjs:1.0 .

.PHONY: run
run:
	xhost + local:root
	sudo docker run -it \
	--env=DISPLAY=$(DISPLAY) \
	--env=QT_X11_NO_MITSHM=1 \
	--network=host \
	--privileged \
	-v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	pyhjs:1.0 /bin/bash

