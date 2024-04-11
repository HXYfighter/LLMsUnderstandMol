import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

import guacamol
from guacamol.scoring_function import ScoringFunction
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.utils.chemistry import canonicalize_list

from model import GPT, GPTConfig
from vocabulary import read_vocabulary, SMILESTokenizer
from utils import set_seed, sample_SMILES, likelihood, to_tensor

from learner import learn_SPE
from tokenizer import SPE_Tokenizer

# Target SMILES
Celecoxib = ["Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1", \
            "c1(-c2ccc(C)cc2)n(-c2ccc(S(N)(=O)=O)cc2)nc(C(F)(F)F)c1", \
            "c1c(S(N)(=O)=O)ccc(-n2nc(C(F)(F)F)cc2-c2ccc(C)cc2)c1", \
            "c1(-n2nc(C(F)(F)F)cc2-c2ccc(C)cc2)ccc(S(=O)(=O)N)cc1", \
            "c1(C(F)(F)F)cc(-c2ccc(C)cc2)n(-c2ccc(S(=O)(N)=O)cc2)n1"]
Troglitazone = ["Cc1c(C)c2c(c(C)c1O)CCC(C)(COc1ccc(CC3SC(=O)NC3=O)cc1)O2", \
                "CC1(COc2ccc(CC3C(=O)NC(=O)S3)cc2)Oc2c(C)c(C)c(O)c(C)c2CC1", \
                "c12c(c(C)c(O)c(C)c1C)CCC(C)(COc1ccc(CC3C(=O)NC(=O)S3)cc1)O2", \
                "C1(COc2ccc(CC3C(=O)NC(=O)S3)cc2)(C)CCc2c(C)c(O)c(C)c(C)c2O1", \
                "c1(C)c2c(c(C)c(C)c1O)OC(C)(COc1ccc(CC3SC(=O)NC3=O)cc1)CC2"]
Thiothixene = ["CN1CCN(CC/C=C2/c3ccccc3Sc3ccc(S(=O)(=O)N(C)C)cc32)CC1", \
                "c1cc2c(cc1)Sc1c(cc(S(=O)(=O)N(C)C)cc1)/C2=C\CCN1CCN(C)CC1", \
                "c1cc2c(cc1)/C(=C/CCN1CCN(C)CC1)c1cc(S(=O)(N(C)C)=O)ccc1S2", \
                "C(N1CCN(C)CC1)C/C=C1/c2ccccc2Sc2c1cc(S(N(C)C)(=O)=O)cc2", \
                "c1c(S(=O)(=O)N(C)C)ccc2c1/C(=C\CCN1CCN(C)CC1)c1ccccc1S2"]

class generator_guacamol(GoalDirectedGenerator):

    def __init__(self, logger, configs):
        self.writer = logger
        self.model_type = configs.model_type
        self.prior_path = configs.prior_path
        self.task_id = configs.guacamol_id
        self.batch_size = configs.batch_size
        self.n_steps = configs.n_steps
        self.learning_rate = configs.learning_rate
        self.sigma = configs.sigma
        self.memory = pd.DataFrame(columns=["smiles", "scores"])
        self.memory_size = configs.memory_size

        self.vocab_path = configs.vocab_path
        self.voc = read_vocabulary(self.vocab_path)
        self.tokenizer = SMILESTokenizer()

        if self.model_type == "gpt":
            prior_config = GPTConfig(self.voc.__len__(), n_layer=8, n_head=8, n_embd=256, block_size=128)
            self.prior = GPT(prior_config).to("cuda")
            self.agent = GPT(prior_config).to("cuda")
            self.optimizer = self.agent.configure_optimizers(weight_decay=0.1, learning_rate=self.learning_rate, betas=(0.9, 0.95))
        else:
            Exception("Undefined model type!")

        self.prior.load_state_dict(torch.load(self.prior_path), strict=True)
        for param in self.prior.parameters():
            param.requires_grad = False
        self.prior.eval()
        self.agent.load_state_dict(torch.load(self.prior_path), strict=True)


    def generate_optimized_molecules(self, scoring_function, number_molecules, starting_population=None):

        for step in tqdm(range(self.n_steps)):
            samples, seqs, _ = sample_SMILES(self.agent, self.voc, n_mols=self.batch_size)
            if self.model_type == "rnn":
                self.agent.train()
            # print(seqs.shape, seqs[0])
            scores = scoring_function.score_list(samples)
            self._memory_update(samples, scores)

            # SMILES Pair Encoding Analysis
            Target_mol = Celecoxib

            outfile = open(f"guacamol_{self.task_id}/tokens_step{step}.txt", 'w')
            fragments = learn_SPE(samples, outfile, num_symbols=1000, min_frequency=200, augmentation=10)
            outfile.close()
            self.writer.add_scalar('high-freq fragments', len(fragments), step)

            codes = open(f"guacamol_{self.task_id}/tokens_step{step}.txt", 'r')
            SPE_tokenizer = SPE_Tokenizer(codes)
            codes.close()
            prob = likelihood(self.agent, smiles=Target_mol, tokenizer=self.tokenizer, vocab=self.voc)
            for i in range(len(Target_mol)):
                self.writer.add_scalars('Target likelihood', {Target_mol[i]: -prob[i]}, step)
                tokens = SPE_tokenizer.tokenize(Target_mol[i])
                self.writer.add_scalars('Number of tokens', {Target_mol[i]: len(tokens)}, step)

            prior_likelihood = likelihood(self.prior, seqs=seqs)
            agent_likelihood = likelihood(self.agent, seqs=seqs)
            loss = torch.pow(self.sigma * to_tensor(np.array(scores)) - (prior_likelihood - agent_likelihood), 2)
            
            loss = loss.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # tensorboard
            self.writer.add_scalar('training loss', loss.item(), step)
            self.writer.add_scalar('mean score', np.mean(scores), step)
            self.writer.add_scalar('loss diff', torch.mean(prior_likelihood - agent_likelihood).item(), step)
            
            if self.task_id in list(range(0, 3)):
                self.writer.add_scalar('top-1 score', np.max(np.array(self.memory["scores"])), step)

            self.writer.add_scalar('mean score in memory', np.mean(np.array(self.memory["scores"])), step)

        samples_all = canonicalize_list(list(self.memory['smiles']))
        scores_all = scoring_function.score_list(samples_all)
        scored_molecules = zip(samples_all, scores_all)
        assert len(samples_all) == len(scores_all)
        sorted_scored_molecules = sorted(scored_molecules, key=lambda x: (x[1], hash(x[0])), reverse=True)
        top_scored_molecules = sorted_scored_molecules[:number_molecules]

        return [x[0] for x in top_scored_molecules]


    def _memory_update(self, smiles, scores):
        scores = list(scores)
        for i in range(len(smiles)):
            if scores[i] < 0 or smiles[i] == "":
                continue
            # canonicalized SMILES and fingerprints
            smiles_i = canonicalize_list([smiles[i]])
            new_data = pd.DataFrame({"smiles": smiles_i[0], "scores": scores[i]}, index=[0])
            self.memory = pd.concat([self.memory, new_data], ignore_index=True, sort=False)

        self.memory = self.memory.drop_duplicates(subset=["smiles"])
        self.memory = self.memory.sort_values('scores', ascending=False)
        self.memory = self.memory.reset_index(drop=True)
        if len(self.memory) > self.memory_size:
            self.memory = self.memory.head(self.memory_size)
