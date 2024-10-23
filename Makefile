build:
	docker build -t cacophony:aframires .
run:
	docker run -it --gpus all --rm -v $$PWD:/home/ubuntu/ -v /home/ubuntu/beta_loops:/home/ubuntu/beta_loops cacophony:aframires