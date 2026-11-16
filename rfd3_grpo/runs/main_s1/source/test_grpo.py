import unittest

import torch

from .grpo import NoiseLoRA,group_advantages,gaussian_logprob,log_ratio,transition_loss


class GRPOTests(unittest.TestCase):
    def test_group_normalization(self):
        r=torch.tensor([[1.,2.,3.],[100.,110.,120.]])
        adv=group_advantages(r)
        torch.testing.assert_close(adv.mean(-1),torch.zeros(2),atol=1e-6,rtol=0)
        torch.testing.assert_close(adv.std(-1,correction=0),torch.ones(2))
        torch.testing.assert_close(adv[0],adv[1])

    def test_ties_have_no_fabricated_signal(self):
        torch.testing.assert_close(group_advantages(torch.ones(2,4)),torch.zeros(2,4))

    def test_conditions_do_not_mix(self):
        r=torch.tensor([[1.,3.],[100.,104.]])
        before=group_advantages(r)[0]
        r[1]*=100
        torch.testing.assert_close(group_advantages(r)[0],before)

    def test_stable_ratio_is_exact_joint_gaussian_ratio(self):
        torch.manual_seed(8)
        x,old,new=torch.randn(3,8,3,dtype=torch.float64),torch.randn(3,8,3,dtype=torch.float64),torch.randn(3,8,3,dtype=torch.float64)
        expected=gaussian_logprob(x,new,.4)-gaussian_logprob(x,old,.4)
        torch.testing.assert_close(log_ratio(x,old,new,.4),expected)

    def test_zero_initialization_matches_reference(self):
        adapter=NoiseLoRA(12,rank=2)
        x=torch.randn(1,8,12)
        self.assertEqual(adapter(x,4).abs().max().item(),0)
        self.assertEqual(sum(p.numel() for p in adapter.parameters()),3*2*(12+3))

    def test_low_rank_projection_matches_merged_weight(self):
        adapter=NoiseLoRA(12,rank=2)
        with torch.no_grad():adapter.b.normal_(0,.1)
        x=torch.randn(2,8,12)
        expected=torch.nn.functional.linear(x,adapter.delta_weight(4))
        torch.testing.assert_close(adapter(x,4),expected)

    def test_positive_advantage_increases_recorded_action_probability(self):
        torch.manual_seed(77)
        adapter=NoiseLoRA(12,rank=2)
        features=torch.randn(1,8,12)
        old=torch.zeros(1,8,3)
        action=torch.randn_like(old)*.2
        record=dict(features=features,ref_mean=old,old_mean=old,action=action,
                    std=.2,coefficient=.8,sigma_hat=4.)
        opt=torch.optim.SGD(adapter.parameters(),lr=.1)
        loss,_=transition_loss(record,adapter,torch.tensor(1.),kl_beta=0)
        loss.backward()
        self.assertGreater(adapter.b.grad[1].abs().sum().item(),0)
        self.assertEqual(adapter.b.grad[0].abs().sum().item(),0)
        opt.step()
        self.assertGreater(log_ratio(action,old,.8*adapter(features,4),.2).item(),0)


if __name__=="__main__":unittest.main()
