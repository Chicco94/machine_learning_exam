#from models.train_model import train_model
from models.test_model import test_model
#from visualization.visualize import plot_scores

def main():
	#train_model(100_000,model_name='breakout_model_96900')
	test_model(start=97000,count=1,render=True,num_eps=1)
	#with open('../reports/log.log','r') as data:
	#	models,scores,steps_lengths = [],[],[]
	#	for line in data.readlines():
	#		line.replace('\n','')
	#		model,score,step_legth = line.split(';')
	#		models.append(model)
	#		scores.append(float(score))
	#		steps_lengths.append(float(step_legth))
	#	plot_scores(scores)



if __name__=='__main__':
    main()