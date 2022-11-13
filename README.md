
Run		svm(Test accuracy)		decision_tree(test accuracy)

 1 		0.986637 					0.828508
 2 		0.980501					0.827298 
 3 		0.986072  					0.832869
 4 		0.994444  					0.855556 
 5 		0.988827  					0.882682 
 
 mean 		0.987296 					0.845382
 std 		0.005038  					0.023786 
 
 
 '''
 docker build -t exp:v1 -f docker/Dockerfile .
 docker run -it exp:v1
 '''
 
 '''
 export FLASK_APP=api/app.py ; flask run
 '''
