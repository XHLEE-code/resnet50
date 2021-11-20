import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from resnet import resnet50, resnet18


class Normalize(nn.Module):
	def __init__(self, power=2):
		super(Normalize, self).__init__()
		self.power = power
	
	def forward(self, x):
		norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
		out = x.div(norm)
		return out


class Non_local(nn.Module):
	def __init__(self, in_channels, reduc_ratio=2):
		super(Non_local, self).__init__()
		
		self.in_channels = in_channels
		self.inter_channels = reduc_ratio // reduc_ratio
		
		self.g = nn.Sequential(
			nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
			          padding=0),
		)
		
		self.W = nn.Sequential(
			nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
			          kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(self.in_channels),
		)
		nn.init.constant_(self.W[1].weight, 0.0)
		nn.init.constant_(self.W[1].bias, 0.0)
		
		self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
		                       kernel_size=1, stride=1, padding=0)
		
		self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
		                     kernel_size=1, stride=1, padding=0)
	
	def forward(self, x):
		'''
                :param x: (b, c, t, h, w)
                :return:
                '''
		
		batch_size = x.size(0)
		g_x = self.g(x).view(batch_size, self.inter_channels, -1)
		g_x = g_x.permute(0, 2, 1)
		
		theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
		theta_x = theta_x.permute(0, 2, 1)
		phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
		f = torch.matmul(theta_x, phi_x)
		N = f.size(-1)
		# f_div_C = torch.nn.functional.softmax(f, dim=-1)
		f_div_C = f / N
		
		y = torch.matmul(f_div_C, g_x)
		y = y.permute(0, 2, 1).contiguous()
		y = y.view(batch_size, self.inter_channels, *x.size()[2:])
		W_y = self.W(y)
		z = W_y + x
		
		return z


# #####################################################################
def weights_init_kaiming(m):
	classname = m.__class__.__name__
	# print(classname)
	if classname.find('Conv') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
	elif classname.find('Linear') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
		init.zeros_(m.bias.data)
	elif classname.find('BatchNorm1d') != -1:
		init.normal_(m.weight.data, 1.0, 0.01)
		init.zeros_(m.bias.data)


def weights_init_classifier(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		init.normal_(m.weight.data, 0, 0.001)
		if m.bias:
			init.zeros_(m.bias.data)


class visible_module(nn.Module):
	def __init__(self, arch='resnet50', share_net=1):
		super(visible_module, self).__init__()
		
		model_v = resnet50(pretrained=True,
		                   last_conv_stride=1, last_conv_dilation=1)
		# avg pooling to global pooling
		self.share_net = share_net
		
		if self.share_net == 0:
			pass
		else:
			self.visible = nn.ModuleList()
			self.visible.conv1 = model_v.conv1
			self.visible.bn1 = model_v.bn1
			self.visible.relu = model_v.relu
			self.visible.maxpool = model_v.maxpool
			if self.share_net > 1:
				for i in range(1, self.share_net):
					setattr(self.visible, 'layer' + str(i), getattr(model_v, 'layer' + str(i)))
	
	def forward(self, x):
		if self.share_net == 0:
			return x
		else:
			x = self.visible.conv1(x)
			x = self.visible.bn1(x)
			x = self.visible.relu(x)
			x = self.visible.maxpool(x)
			
			if self.share_net > 1:
				for i in range(1, self.share_net):
					x = getattr(self.visible, 'layer' + str(i))(x)
			return x


class rgbir_module(nn.Module):
	def __init__(self, arch='resnet50', share_net=1):
		super(rgbir_module, self).__init__()
		
		model_v = resnet50(pretrained=True,
		                   last_conv_stride=1, last_conv_dilation=1)
		# avg pooling to global pooling
		self.share_net = share_net
		
		if self.share_net == 0:
			pass
		else:
			self.rgbir = nn.ModuleList()
			self.rgbir.conv1 = model_v.conv1
			self.rgbir.bn1 = model_v.bn1
			self.rgbir.relu = model_v.relu
			self.rgbir.maxpool = model_v.maxpool
			if self.share_net > 1:
				for i in range(1, self.share_net):
					setattr(self.rgbir, 'layer' + str(i), getattr(model_v, 'layer' + str(i)))
	
	def forward(self, x):
		if self.share_net == 0:
			return x
		else:
			x = self.rgbir.conv1(x)
			x = self.rgbir.bn1(x)
			x = self.rgbir.relu(x)
			x = self.rgbir.maxpool(x)
			
			if self.share_net > 1:
				for i in range(1, self.share_net):
					x = getattr(self.rgbir, 'layer' + str(i))(x)
			return x


class thermal_module(nn.Module):
	def __init__(self, arch='resnet50', share_net=1):
		super(thermal_module, self).__init__()
		
		model_t = resnet50(pretrained=True,
		                   last_conv_stride=1, last_conv_dilation=1)
		# avg pooling to global pooling
		self.share_net = share_net
		
		if self.share_net == 0:
			pass
		else:
			self.thermal = nn.ModuleList()
			self.thermal.conv1 = model_t.conv1
			self.thermal.bn1 = model_t.bn1
			self.thermal.relu = model_t.relu
			self.thermal.maxpool = model_t.maxpool
			if self.share_net > 1:
				for i in range(1, self.share_net):
					setattr(self.thermal, 'layer' + str(i), getattr(model_t, 'layer' + str(i)))
	
	def forward(self, x):
		if self.share_net == 0:
			return x
		else:
			x = self.thermal.conv1(x)
			x = self.thermal.bn1(x)
			x = self.thermal.relu(x)
			x = self.thermal.maxpool(x)
			
			if self.share_net > 1:
				for i in range(1, self.share_net):
					x = getattr(self.thermal, 'layer' + str(i))(x)
			return x


class base_resnet(nn.Module):
	def __init__(self, arch='resnet50', share_net=1):
		super(base_resnet, self).__init__()
		
		model_base = resnet50(pretrained=True,
		                      last_conv_stride=1, last_conv_dilation=1)
		# avg pooling to global pooling
		model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.share_net = share_net
		if self.share_net == 0:
			self.base = model_base
		else:
			self.base = nn.ModuleList()
			
			if self.share_net > 4:
				pass
			else:
				for i in range(self.share_net, 5):
					setattr(self.base, 'layer' + str(i), getattr(model_base, 'layer' + str(i)))
	
	def forward(self, x):
		if self.share_net == 0:
			x = self.base.conv1(x)
			x = self.base.bn1(x)
			x = self.base.relu(x)
			x = self.base.maxpool(x)
			
			x = self.base.layer1(x)
			x = self.base.layer2(x)
			x = self.base.layer3(x)
			x = self.base.layer4(x)
			return x
		elif self.share_net > 4:
			return x
		else:
			for i in range(self.share_net, 5):
				x = getattr(self.base, 'layer' + str(i))(x)
			return x


class embed_net(nn.Module):
	def __init__(self, class_num, gm_pool='on', arch='resnet50', share_net=1, mgn='on', local_feat_dim=256, num_strips_h=3):
		super(embed_net, self).__init__()
		
		self.thermal_module = thermal_module(arch=arch, share_net=share_net)
		self.visible_module = visible_module(arch=arch, share_net=share_net)
		self.rgbir_module = rgbir_module(arch=arch, share_net=share_net)
		self.base_resnet = base_resnet(arch=arch, share_net=share_net)
		
		self.mgn = mgn
		pool_dim = 2048
		self.l2norm = Normalize(2)
		self.gm_pool = gm_pool
		
		if self.mgn == 'on':
			self.num_stripes_h = num_strips_h  # 6
			local_conv_out_channels = local_feat_dim  # 256
			
			self.local_conv_list_c = nn.ModuleList()
			conv = nn.Conv2d(pool_dim, local_conv_out_channels, 1)  # 2048 256
			conv.apply(weights_init_kaiming)
			self.local_conv_list_c.append(nn.Sequential(
				conv,
				nn.BatchNorm2d(local_conv_out_channels),
				nn.ReLU(inplace=True)
			))
			
			self.fc_list_c = nn.ModuleList()
			fc = nn.Linear(local_conv_out_channels, class_num)  # 256 395
			init.normal_(fc.weight, std=0.001)
			init.constant_(fc.bias, 0)
			self.fc_list_c.append(fc)
			
			self.local_conv_list_h = nn.ModuleList()
			for _ in range(self.num_stripes_h):
				conv = nn.Conv2d(pool_dim, local_conv_out_channels, 1)  # 2048 256
				conv.apply(weights_init_kaiming)
				self.local_conv_list_h.append(nn.Sequential(
					conv,
					nn.BatchNorm2d(local_conv_out_channels),
					nn.ReLU(inplace=True)
				))
			
			self.fc_list_h = nn.ModuleList()
			for _ in range(self.num_stripes_h):
				fc = nn.Linear(local_conv_out_channels, class_num)  # 256 395
				init.normal_(fc.weight, std=0.001)
				init.constant_(fc.bias, 0)
				self.fc_list_h.append(fc)
		
		
		else:
			self.bottleneck = nn.BatchNorm1d(pool_dim)
			self.bottleneck.bias.requires_grad_(False)  # no shift
			
			self.classifier = nn.Linear(pool_dim, class_num, bias=False)
			
			self.bottleneck.apply(weights_init_kaiming)
			self.classifier.apply(weights_init_classifier)
			self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
	
	def forward(self, x1, x3, x2, modal=0):  # x1:torch.Size([32, 3, 288, 144])    x2:torch.Size([32, 3, 288, 144])
		if modal == 0:
			x1 = self.visible_module(x1)  # torch.Size([64, 64, 72, 36])
			x3 = self.rgbir_module(x3)  # torch.Size([64, 64, 72, 36])
			x2 = self.thermal_module(x2)  # torch.Size([64, 64, 72, 36])
			x = torch.cat((x1, x3, x2), 0)  # torch.Size([192, 64, 72, 36])
		elif modal == 1:
			x = self.visible_module(x1)
		elif modal == 2:
			x = self.thermal_module(x2)
		
		# shared block
		x = self.base_resnet(x)                      # torch.Size([192, 2048, 18, 9])
		
		
		if self.mgn == 'on':
			
			feat_c = x
			whole_feat_c_list = []
			class_feat_c_list = []
			if self.gm_pool == 'on':
				b, c, h, w = feat_c.shape
				feat_c = feat_c.view(b, c, -1)                                 # torch.Size([192, 2048, 162])
				p = 3.0
				feat_c = (torch.mean(feat_c ** p, dim=-1) + 1e-12) ** (1 / p)  # torch.Size([192, 2048])
			else:
				feat_c = self.avgpool(feat_c)
				feat_c = feat_c.view(feat_c.size(0), feat_c.size(1))
			
			whole_feat_c = self.local_conv_list_c[0](
				feat_c.view(feat_c.size(0), feat_c.size(1), 1, 1))      # torch.Size([192, 256, 1, 1])
			whole_feat_c = whole_feat_c.view(whole_feat_c.size(0), -1)  # torch.Size([192, 256])
			whole_feat_c_list.append(whole_feat_c)
			
			if hasattr(self, 'fc_list_c'):
				class_feat_c_list.append(self.fc_list_c[0](whole_feat_c))
			
			feat_all_c = [lf for lf in whole_feat_c_list]
			feat_all_c = torch.cat(feat_all_c, dim=1)  # torch.Size([192, 256])
			
			
			feat_h = x                                           # torch.Size([64, 2048, 18, 9])
			assert feat_h.size(2) % self.num_stripes_h == 0
			stripe_h = int(feat_h.size(2) / self.num_stripes_h)  # 18/6 = 3
			local_feat_h_list = []
			class_feat_h_list = []
			for i in range(self.num_stripes_h):
				
				if self.gm_pool == 'on':
					# gm pool
					local_feat_h = feat_h[:, :, i * stripe_h: (i + 1) * stripe_h, :]  # torch.Size([128, 2048, 3, 9])
					b, c, h, w = local_feat_h.shape
					local_feat_h = local_feat_h.view(b, c, -1)  # torch.Size([128, 2048, 27])
					p = 3.0
					local_feat_h = (torch.mean(local_feat_h ** p, dim=-1) + 1e-12) ** (
							1 / p)  # torch.Size([128, 2048])
				else:
					local_feat_h = F.max_pool2d(feat_h[:, :, i * stripe_h: (i + 1) * stripe_h, :],
					                            (stripe_h, feat_h.size(-1)))
				
				local_feat_h = self.local_conv_list_h[i](
					local_feat_h.view(feat_h.size(0), feat_h.size(1), 1, 1))  # torch.Size([128, 256, 1, 1])
				
				local_feat_h = local_feat_h.view(local_feat_h.size(0), -1)
				local_feat_h_list.append(local_feat_h)  # torch.Size([128, 256])
				
				if hasattr(self, 'fc_list_h'):
					class_feat_h_list.append(self.fc_list_h[i](local_feat_h))  # torch.Size([128, 395])
			
			feat_all_h = [lf for lf in local_feat_h_list]
			feat_all_h = torch.cat(feat_all_h, dim=1)              # torch.Size([128, 1536])
			
			feat_all = torch.cat((feat_all_c, feat_all_h), dim=1)  # torch.Size([128, 2560])
			
			if self.training:
				return whole_feat_c_list, class_feat_c_list, local_feat_h_list, class_feat_h_list, feat_all
			else:
				return self.l2norm(feat_all)
		else:
			if self.gm_pool == 'on':
				b, c, h, w = x.shape
				x = x.view(b, c, -1)
				p = 3.0
				x_pool = (torch.mean(x ** p, dim=-1) + 1e-12) ** (1 / p)
			else:
				x_pool = self.avgpool(x)
				x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))
			
			feat = self.bottleneck(x_pool)
			
			if self.training:
				return x_pool, self.classifier(feat)
			else:
				return self.l2norm(x_pool), self.l2norm(feat)




if __name__ == "__main__":
	x = torch.randn(144, 2048, 18, 9)
	net = CA(2048)
	b = net(x)
	print(b.shape)


