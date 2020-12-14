import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class OTAM_Loss(nn.Module):
	def __init__(self, smooth_param=0.1, sim_metric='cosine', relaxation=True):
		super(OTAM_Loss, self).__init__()
		self.lam = smooth_param
		self.sim_metric = sim_metric
		self.criterion = nn.CrossEntropyLoss()
		self.relaxation=relaxation

	def forward(self, supp, query, query_ys):
		## supp: B*N*T*D, query: B*M*T*D
		B, _, T, D = supp.shape
		num_supp = supp.shape[1]
		num_query = query.shape[1]
		assert T == query.shape[2]

		supp = supp.unsqueeze(1).repeat((1,num_query,1,1,1)).view(-1, T, D)
		query = query.unsqueeze(2).repeat((1,1,num_supp,1,1)).view(-1, T, D)

		
		supp = supp.unsqueeze(2).repeat((1,1,T,1))
		query = query.unsqueeze(1).repeat((1,T,1,1))
		
		dist = self.calculate_dist(supp, query)
		tam1 = self.dtw(dist, self.relaxation).view(-1, num_supp)

		dist = torch.flip(dist, (1,2))
		tam2 = self.dtw(dist, self.relaxation).view(-1, num_supp)

		loss1 = self.criterion(-tam1, query_ys)
		loss2 = self.criterion(-tam2, query_ys)

		loss = (loss1 + loss2) / 2
		tam = (tam1 + tam2) / 2
		return loss, tam


	def dtw(self, dist, relaxation=True):
		T = dist.shape[-1]
		dist = F.pad(dist, (1,1,0,0), 'constant', 0.0)
		opt = torch.zeros_like(dist)
		
		assert dist.shape == opt.shape

		for l in range(0, T):
			for m in range(1, T+2):
				c1 = - opt[:, l-1, m-1] / self.lam
				c2 = - opt[:, l, m-1] / self.lam
				c3 = - opt[:, l-1, m] / self.lam
				if l == 0:
					opt[:,l,m] = opt[:, l, m-1] + dist[:,l,m]
					continue

				elif m == T+1 or m == 1:
					c = torch.stack([c1,c2,c3],dim=-1) 
				else:
					c = torch.stack([c1,c2],dim=-1) 
				
				if not relaxation:
					opt[:,l,m] = torch.min(-c*self.lam, dim=-1)[0]
				else:
					opt[:,l,m] = -self.lam * (torch.log(torch.exp(c).sum(dim=-1)))		
				opt[:,l,m] += dist[:,l,m]

		return opt[:, -1, -1]

	
	def calculate_dist(self, supp, query):
		if self.sim_metric == 'cosine':
			dist = 1 - F.cosine_similarity(supp, query, dim=-1)
			return dist


class Baseline_Loss(nn.Module):
	def __init__(self, sim_metric='cosine'):
		super(Baseline_Loss, self).__init__()
		self.sim_metric = sim_metric
		self.criterion = nn.CrossEntropyLoss()

	def forward(self, supp, query, query_ys):
		## supp: B*N*T*D, query: B*M*T*D
		B, _, T, D = supp.shape
		num_supp = supp.shape[1]
		num_query = query.shape[1]
		assert T == query.shape[2]

		supp = supp.unsqueeze(1).repeat((1,num_query,1,1,1))
		query = query.unsqueeze(2).repeat((1,1,num_supp,1,1))

		supp = supp.mean(-2)
		query = query.mean(-2)

		dist = F.cosine_similarity(supp, query, dim=-1)
		dist = dist.squeeze(0)

		loss = self.criterion(dist, query_ys)
		return loss, 1 - dist

if __name__ == '__main__':
	#torch.manual_seed(0)
	supp = torch.rand(1,2,3,8)
	query = torch.rand(1,4,3,8)
	#query = supp.repeat((1,2,1,1))
	query_ys = torch.arange(0,2).unsqueeze(1).repeat((1,2)).view(-1)
	otam = OTAM_Loss(smooth_param=0.1)
	loss = otam(supp, query, query_ys)
	print(loss)
