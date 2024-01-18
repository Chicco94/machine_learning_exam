#from models.train_model import train_model
from models.test_model import test_model, test_single_model
from visualization.visualize import plot_scores,np

def main():
	#train_model(100_000,model_name='breakout_model_96900')
	#test_model(start=73500,end=100_000,step=100,render=True,num_eps=1,model_name_struct='breakout_model_')
	test_single_model('breakout_model_169700')
	#with open('../reports/log3.log','r') as data:
	#	models,scores,steps_lengths = [],[],[]
	#	for line in data.readlines():
	#		line.replace('\n','')
	#		model,score,step_legth = line.split(';')
	#		score = float(score)
	#		models.append(model)
	#		scores.append(float(score))
	#		#steps_lengths.append(float(step_legth))
#
	#	x1 = int(models[0].split('_')[2])
	#	x2 = int(models[-1].split('_')[2])
	#	epochs = np.linspace(x1, x2, len(models))
	#	plot_scores(scores,epochs)



if __name__=='__main__':
	main()