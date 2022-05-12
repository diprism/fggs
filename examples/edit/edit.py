import torch
import fggs
import argparse
import tqdm
import math

class EditDistanceModel(torch.nn.Module):
    def __init__(self, salphabet, talphabet):
        super().__init__()
        self.salphabet = list(salphabet)
        self.talphabet = list(talphabet)
        self.snumberize = {s:i for i,s in enumerate(self.salphabet)}
        self.tnumberize = {t:j for j,t in enumerate(self.talphabet)}
        self.insweight = torch.nn.Parameter(torch.zeros(len(talphabet)))
        self.delweight = torch.nn.Parameter(torch.zeros(len(salphabet)))
        self.subweight = torch.nn.Parameter(torch.zeros(len(salphabet), len(talphabet)))
    
    def make_fgg(self):
        fgg = fggs.FGG("S")

        # Insert
        rhs = fggs.Graph()
        y = rhs.new_node("tsym")
        rhs.new_edge("ins", [y], is_terminal=True)
        rhs.new_edge("S", [], is_nonterminal=True)
        fgg.new_rule("S", rhs)

        # Delete
        rhs = fggs.Graph()
        x = rhs.new_node("ssym")
        rhs.new_edge("del", [x], is_terminal=True)
        rhs.new_edge("S", [], is_nonterminal=True)
        fgg.new_rule("S", rhs)

        # Substitute
        rhs = fggs.Graph()
        x = rhs.new_node("ssym")
        y = rhs.new_node("tsym")
        rhs.new_edge("sub", [x, y], is_terminal=True)
        rhs.new_edge("S", [], is_nonterminal=True)
        fgg.new_rule("S", rhs)

        # End
        rhs = fggs.Graph()
        fgg.new_rule("S", rhs)

        fgg.new_finite_domain("ssym", self.salphabet)
        fgg.new_finite_domain("tsym", self.talphabet)
        fgg.new_finite_factor("ins", self.insweight)
        fgg.new_finite_factor("del", self.delweight)
        fgg.new_finite_factor("sub", self.subweight)
        return fgg
    
    def make_fgg_s(self, sstring):
        m = len(sstring)
        fgg = fggs.FGG("S-0")

        for i in range(m+1):
            # Insert
            rhs = fggs.Graph()
            y = rhs.new_node("tsym")
            rhs.new_edge("ins", [y], is_terminal=True)
            rhs.new_edge(f"S-{i}", [], is_nonterminal=True)
            fgg.new_rule(f"S-{i}", rhs)

        for i, s in enumerate(sstring):
            # Delete
            rhs = fggs.Graph()
            rhs.new_edge(f"del-{s}", [], is_terminal=True)
            rhs.new_edge(f"S-{i+1}", [], is_nonterminal=True)
            fgg.new_rule(f"S-{i}", rhs)

            # Substitute
            rhs = fggs.Graph()
            y = rhs.new_node("tsym")
            rhs.new_edge(f"sub-{s}", [y], is_terminal=True)
            rhs.new_edge(f"S-{i+1}", [], is_nonterminal=True)
            fgg.new_rule(f"S-{i}", rhs)

        # End
        rhs = fggs.Graph()
        fgg.new_rule(f"S-{m}", rhs)

        fgg.new_finite_domain("tsym", self.talphabet)
        fgg.new_finite_factor("ins", self.insweight)
        for s in set(sstring):
            snum = self.snumberize[s]
            fgg.new_finite_factor(f"del-{s}", self.delweight[snum])
            fgg.new_finite_factor(f"sub-{s}", self.subweight[snum])
        return fgg

    def make_fgg_st(self, sstring, tstring):
        m = len(sstring)
        n = len(tstring)
        fgg = fggs.FGG("S-0-0")

        for i in range(m+1):
            for j, t in enumerate(tstring):
                # Insert
                rhs = fggs.Graph()
                rhs.new_edge("ins-{t}", [], is_terminal=True)
                rhs.new_edge(f"S-{i}-{j+1}", [], is_nonterminal=True)
                fgg.new_rule(f"S-{i}-{j}", rhs)

        for i, s in enumerate(sstring):
            for j in range(n+1):
                # Delete
                rhs = fggs.Graph()
                rhs.new_edge(f"del-{s}", [], is_terminal=True)
                rhs.new_edge(f"S-{i+1}-{j}", [], is_nonterminal=True)
                fgg.new_rule(f"S-{i}-{j}", rhs)

        for i, s in enumerate(sstring):
            for j, t in enumerate(tstring):
                # Substitute
                rhs = fggs.Graph()
                rhs.new_edge(f"sub-{s}-{t}", [], is_terminal=True)
                rhs.new_edge(f"S-{i+1}-{j+1}", [], is_nonterminal=True)
                fgg.new_rule(f"S-{i}-{j}", rhs)

        # End
        rhs = fggs.Graph()
        fgg.new_rule(f"S-{m}-{n}", rhs)

        for t in set(tstring):
            tnum = self.tnumberize[t]
            fgg.new_finite_factor("ins-{t}", self.insweight[tnum])
        for s in set(sstring):
            snum = self.snumberize[s]
            fgg.new_finite_factor(f"del-{s}", self.delweight[snum])
            for t in set(tstring):
                tnum = self.tnumberize[t]
                fgg.new_finite_factor(f"sub-{s}-{t}", self.subweight[snum,tnum])
        return fgg

if __name__ == "__main__":    
    ap = argparse.ArgumentParser()
    ap.add_argument('trainfile')
    ap.add_argument('--device', dest="device", default="cpu")
    args = ap.parse_args()

    salphabet = set()
    talphabet = set()
    traindata = []
    for line in open(args.trainfile):
        sstring, tstring = line.rstrip().split('\t')
        salphabet.update(sstring)
        talphabet.update(tstring)
        traindata.append((list(sstring), list(tstring)))

    semiring = fggs.LogSemiring(device=args.device)
    m = EditDistanceModel(salphabet, talphabet)
    o = torch.optim.Adam(m.parameters(), lr=0.05)
    for epoch in range(100):
        train_loss = 0.
        for sstring, tstring in tqdm.tqdm(traindata):
            fgg_s = m.make_fgg_s(sstring)
            z_s = fggs.sum_product(fgg_s, method='newton', semiring=semiring)
            fgg_st = m.make_fgg_st(sstring, tstring)
            z_st = fggs.sum_product(fgg_st, method='newton', semiring=semiring)
            loss = -(z_st - z_s)
            o.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(m.parameters(), 10.)
            o.step()
            #print('ins', m.insweight)
            #print('del', m.delweight)
            #print('sub', m.subweight)
            train_loss += loss.item()
        print(f'epoch={epoch+1} train_loss={train_loss}', flush=True)
