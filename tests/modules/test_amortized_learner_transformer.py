import numpy as np
import torch
from alef.active_learners.amortized_policies.nn.modules.selfattention import (
    SelfAttention,
    DoubleSequencesSelfAttention,
)
from alef.active_learners.amortized_policies.nn.policies import (
    ContinuousGPPolicy,
    ContinuousGPBudgetParsedPolicy,
    SafetyAwareContinuousGPPolicy,
    SafetyAwareContinuousGPBudgetParsedPolicy,
)
from alef.active_learners.amortized_policies.nn.policies_flex_dim import (
    ContinuousGPFlexDimPolicy,
    ContinuousGPFlexDimBudgetParsedPolicy,
    SafetyAwareContinuousGPFlexDimPolicy,
    SafetyAwareContinuousGPFlexDimBudgetParsedPolicy,
)

B = 10
N1 = 20
N2 = 30
E = 32
D = 2

def get_mask(batch_size, seq_len):
    """
    :param batch_size: int
    :param seq_len: int
    
    :return [batch_size, 1], [batch_size, seq_len]
    """
    seq_count = torch.randint(1, seq_len+1, size=[batch_size]).unsqueeze(-1)
    mask = torch.where(
        torch.arange(0, seq_len, dtype=int).expand( [batch_size, seq_len] ) < seq_count,
        torch.ones([batch_size, seq_len], dtype=int),
        torch.zeros([batch_size, seq_len], dtype=int),
    )
    return seq_count, mask

def test_transformer_mask():
    x = torch.randn(
        [B, N1, E],
        dtype=torch.get_default_dtype()
    )
    seq_count, mask = get_mask(B, N1)
    model = SelfAttention(E, E, n_attn_layers=2)
    y_masked = model.forward(x, mask=mask) # [B, N, E]
    for b in range(B):
        yb = model.forward(x[b,None,:seq_count[b,0]]) # [1, N, E]
        assert torch.allclose(yb, y_masked[ b, None, :seq_count[b,0] ], atol=1e-5)

def test_two_seqs_transformer_mask():
    x = torch.randn(
        [B, N2, N1, E],
        dtype=torch.get_default_dtype()
    )
    seq1_count, mask1 = get_mask(B, N1)
    seq2_count, mask2 = get_mask(B, N2)
    model = DoubleSequencesSelfAttention(E, E, n_attn_layers=2)
    y_masked = model.forward(x, mask1=mask1, mask2=mask2) # [B, N2, N1, E]
    for b in range(B):
        l2 = seq2_count[b,0]
        l1 = seq1_count[b,0]
        yb = model.forward(x[b,None,:l2,:l1]) # [1, N, E]
        assert torch.allclose(yb, y_masked[ b, None, :l2, :l1 ], atol=1e-5)

### now test the policies

def test_al_policy_mask():
    with torch.no_grad():
        X = torch.randn( [B, N1, D], dtype=torch.get_default_dtype() )
        Y = torch.randn( [B, N1], dtype=torch.get_default_dtype() )
        budget = torch.ones( [B, 1], dtype=torch.get_default_dtype() )
        seq_count, mask = get_mask(B, N1)

        ##
        policy = ContinuousGPPolicy(D, 1, hidden_dim_encoder=[512], encoding_dim=E, hidden_dim_emitter=[512])
        query = policy(X, Y, mask_sequence=mask) # [B, 1, D]
        for b in range(B):
            l = seq_count[b,0]
            qb = policy(X[b, :l], Y[b, :l]) # [1, D]
            assert torch.allclose( qb, query[b], atol=1e-5 ), f'{b}th: {qb} vs {query[b]}'
        ##
        policy = ContinuousGPBudgetParsedPolicy(D, 1, hidden_dim_encoder=[512], encoding_dim=E, hidden_dim_emitter=[512])
        query = policy(budget, X, Y, mask_sequence=mask) # [B, 1, D]
        for b in range(B):
            l = seq_count[b,0]
            qb = policy(budget[b], X[b, :l], Y[b, :l]) # [1, D]
            assert torch.allclose( qb, query[b], atol=1e-5 ), f'{b}th: {qb} vs {query[b]}'

def test_sal_policy_mask():
    with torch.no_grad():
        X = torch.randn( [B, N1, D], dtype=torch.get_default_dtype() )
        Y = torch.randn( [B, N1], dtype=torch.get_default_dtype() )
        Z = torch.randn( [B, N1], dtype=torch.get_default_dtype() )
        budget = torch.ones( [B, 1], dtype=torch.get_default_dtype() )
        seq_count, mask = get_mask(B, N1)

        ##
        policy = SafetyAwareContinuousGPPolicy(D, 1, 1, hidden_dim_encoder=[512], encoding_dim=E, hidden_dim_emitter=[512])
        query = policy(X, Y, Z, mask_sequence=mask) # [B, 1, D]
        for b in range(B):
            l = seq_count[b,0]
            qb = policy(X[b, :l], Y[b, :l], Z[b, :l]) # [1, D]
            assert torch.allclose( qb, query[b], atol=1e-5 ), f'{b}th: {qb} vs {query[b]}'
        ##
        policy = SafetyAwareContinuousGPBudgetParsedPolicy(D, 1, 1, hidden_dim_encoder=[512], encoding_dim=E, hidden_dim_emitter=[512])
        query = policy(budget, X, Y, Z, mask_sequence=mask) # [B, 1, D]
        for b in range(B):
            l = seq_count[b,0]
            qb = policy(budget[b], X[b, :l], Y[b, :l], Z[b, :l]) # [1, D]
            assert torch.allclose( qb, query[b], atol=1e-5 ), f'{b}th: {qb} vs {query[b]}'


### test flexible dimension policies

def test_al_flexdim_policy_mask():
    with torch.no_grad():
        X = torch.randn( [B, N1, D], dtype=torch.get_default_dtype() )
        Y = torch.randn( [B, N1], dtype=torch.get_default_dtype() )
        budget = torch.ones( [B, 1], dtype=torch.get_default_dtype() )
        seq_count1, mask1 = get_mask(B, N1)
        seq_count2, mask2 = get_mask(B, D)

        ##
        policy = ContinuousGPFlexDimPolicy(encoding_dim=E, num_self_attention_layer=2, hidden_dim_emitter=[512])
        query = policy(X, Y, mask_sequence=mask1, mask_feature=mask2) # [B, 1, D]
        for b in range(B):
            l = seq_count1[b,0]
            d = seq_count2[b,0]
            qb = policy(X[b, :l, :d], Y[b, :l]) # [1, D]
            assert torch.allclose( qb, query[b, ..., :d], atol=1e-5 ), f'{b}th: {qb} vs {query[b]}'
        ##
        policy = ContinuousGPFlexDimBudgetParsedPolicy(encoding_dim=E, hidden_dim_budget_encoder=[512], num_self_attention_layer=2, hidden_dim_emitter=[512])
        query = policy(budget, X, Y, mask_sequence=mask1, mask_feature=mask2) # [B, 1, D]
        for b in range(B):
            l = seq_count1[b,0]
            d = seq_count2[b,0]
            qb = policy(budget[b], X[b, :l, :d], Y[b, :l]) # [1, D]
            assert torch.allclose( qb, query[b, ..., :d], atol=1e-5 ), f'{b}th: {qb} vs {query[b]}'

def test_sal_flexdim_policy_mask():
    with torch.no_grad():
        X = torch.randn( [B, N1, D], dtype=torch.get_default_dtype() )
        Y = torch.randn( [B, N1], dtype=torch.get_default_dtype() )
        Z = torch.randn( [B, N1], dtype=torch.get_default_dtype() )
        budget = torch.ones( [B, 1], dtype=torch.get_default_dtype() )
        seq_count1, mask1 = get_mask(B, N1)
        seq_count2, mask2 = get_mask(B, D)

        ##
        policy = SafetyAwareContinuousGPFlexDimPolicy(encoding_dim=E, num_self_attention_layer=2, hidden_dim_emitter=[512])
        query = policy(X, Y, Z, mask_sequence=mask1, mask_feature=mask2) # [B, 1, D]
        for b in range(B):
            l = seq_count1[b,0]
            d = seq_count2[b,0]
            qb = policy(X[b, :l, :d], Y[b, :l], Z[b, :l]) # [1, D]
            assert torch.allclose( qb, query[b, ..., :d], atol=1e-5 ), f'{b}th: {qb} vs {query[b]}'
        ##
        policy = SafetyAwareContinuousGPFlexDimBudgetParsedPolicy(encoding_dim=E, hidden_dim_budget_encoder=[512], num_self_attention_layer=2, hidden_dim_emitter=[512])
        query = policy(budget, X, Y, Z, mask_sequence=mask1, mask_feature=mask2) # [B, 1, D]
        for b in range(B):
            l = seq_count1[b,0]
            d = seq_count2[b,0]
            qb = policy(budget[b], X[b, :l, :d], Y[b, :l], Z[b, :l]) # [1, D]
            assert torch.allclose( qb, query[b, ..., :d], atol=1e-5 ), f'{b}th: {qb} vs {query[b]}'
