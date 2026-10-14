"""Genetic search over discretization schemes.

A chromosome holds one gene per selected numerical attribute. A gene chooses a number of
intervals k (1..b_max) and k-1 cut points from that attribute's candidate pool; k = 1
leaves the attribute continuous. The fitness of a chromosome is the training-set score
of the depth-limited C4.5 tree grown on the data discretized by it.
"""
import copy
import hashlib
import math
import os
import random
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score)

from . import tree
from .candidates import CBS_LOOKAHEAD, build_candidate_bounds


class Gene:
    def __init__(self, feature_name: str, df_id: int, max_disc: int = 3, uniq_vals=None, bounds=None):
        self.feature = feature_name
        self.id = df_id
        self.bin = None
        self.label = None
        self.break_pts = []

        # candidate pool: precomputed (cb / cbs / midpoint) or midpoints of uniq_vals
        if bounds is None:
            bounds = build_candidate_bounds(uniq_vals if uniq_vals is not None else [],
                                            labels=[], mode='midpoint')
        self.bounds = bounds

        # number of intervals; no candidate -> attribute stays continuous
        n_bounds = len(bounds)
        if n_bounds < 1:
            self.disc_num = 1
        else:
            self.disc_num = random.randint(1, min(max_disc, n_bounds))

        # (disc_num - 1) cut points drawn from the pool
        if self.disc_num > 1:
            self.break_pts = sorted(random.sample(bounds, self.disc_num - 1))

    def __repr__(self):
        if self.disc_num == 1:
            return 'Feature ({0}) {1} is not selected'.format(self.id, self.feature)
        return 'Feature ({0}) {1} discretized with {2} classes - break points: {3}'.format(
            self.id, self.feature, self.disc_num, self.break_pts)


def build_feature_value_map(df: pd.DataFrame) -> dict:
    value_map = {}
    for col in df.columns[:-1]:  # all but Decision
        if df[col].dtype == "object":
            value_map[col] = sorted(df[col].dropna().astype(str).unique().tolist())
    return value_map


class Chromosome:
    def __init__(self, attr_names: List[str], df: pd.DataFrame, cat_names: List[str],
                 config=None, target='Decision'):
        self.candidate_mode = (config or {}).get('candidate_mode', 'midpoint')
        # b_max: maximum number of intervals (= branches) per attribute
        self.max_disc = (config or {}).get('max_disc', 4)
        # L: look-ahead window of the CBS filter
        self.lookahead = (config or {}).get('cbs_lookahead', CBS_LOOKAHEAD)
        # candidate pools are computed once per attribute and reused
        self.feature_bounds = {}
        labels = df[target].to_numpy()

        gene_list = []
        exclude_attr = [target]
        for id, attr in enumerate(attr_names):
            if attr not in cat_names and attr not in exclude_attr:
                bounds = build_candidate_bounds(df[attr].to_numpy(), labels, mode=self.candidate_mode,
                                                lookahead=self.lookahead)
                self.feature_bounds[attr] = bounds
                gene_list.append(Gene(feature_name=attr, max_disc=self.max_disc,
                                      df_id=id, bounds=bounds))
        self.genes = gene_list
        self.objective = -1
        self.fitness_value = -1
        self.df = df
        self.target = target
        self.model = None
        self.attr_names = attr_names
        self.cat_names = cat_names
        self.config = copy.deepcopy(config) if config is not None else {
            'enableParallelism': False,
        }

    def clone(self):
        clone_chromosome = Chromosome(self.attr_names, self.df, self.cat_names, config=self.config)
        clone_chromosome.genes = copy.deepcopy(self.genes)
        return clone_chromosome

    def __repr__(self):
        return 'Chromosome with {0} features - Objective {1}'.format(len(self.genes), self.objective)

    def signature(self):
        """Canonical hash of the genotype (used by the fitness cache; consumes no randomness)."""
        sig = tuple((g.feature, g.disc_num,
                     tuple(round(float(b), 12) for b in g.break_pts))
                    for g in self.genes)
        return hashlib.blake2b(repr(sig).encode(), digest_size=8).hexdigest()

    def mutate(self):
        """Redraw one fifth (at least one) of the genes."""
        clone_chromosome = self.clone()
        num_mutations = max(1, len(self.genes) // 5)
        mutation_indices = random.sample(range(len(self.genes)), num_mutations)

        for target_id in mutation_indices:
            target_gene = clone_chromosome.genes[target_id]
            bounds = clone_chromosome.feature_bounds.get(target_gene.feature)
            if bounds is None:
                bounds = build_candidate_bounds(
                    clone_chromosome.df[target_gene.feature].to_numpy(),
                    clone_chromosome.df[clone_chromosome.target].to_numpy(),
                    mode=clone_chromosome.candidate_mode, lookahead=clone_chromosome.lookahead)
            clone_chromosome.genes[target_id] = Gene(feature_name=target_gene.feature,
                                                     max_disc=clone_chromosome.max_disc,
                                                     df_id=target_gene.id, bounds=bounds)
        return clone_chromosome

    def crossover(self, other: 'Chromosome', update_ratio=0.3):
        """Copy ceil(update_ratio * #genes) randomly chosen genes from ``other``."""
        new_chromosome = self.clone()
        num_update = math.ceil(len(new_chromosome.genes) * update_ratio)
        update_gene_ids = random.sample(range(0, len(new_chromosome.genes)), num_update)
        for gene_id in update_gene_ids:
            if gene_id < len(other.genes):
                new_chromosome.genes[gene_id] = copy.deepcopy(other.genes[gene_id])
        return new_chromosome

    def fitness_from(self, evaluation):
        """Fitness used for ranking (larger is better).

        config['fitness_metric']: 'accuracy' (default, chefboost training accuracy 0-100),
        'balanced_accuracy', 'macro_f1', 'mcc' or 'gmean' (computed from chefboost's
        training predictions, no refit).
        """
        metric = (self.config or {}).get('fitness_metric', 'accuracy')
        if metric == 'accuracy':
            return evaluation['Accuracy']

        y_true = evaluation.get('Actuals')
        y_pred = evaluation.get('Predictions')
        if y_true is None or y_pred is None:
            raise RuntimeError(
                "fitness_metric=%r needs training predictions, which chefboost did not return." % metric)
        y_true = [str(v) for v in y_true]
        y_pred = [str(v) for v in y_pred]

        if metric == 'balanced_accuracy':
            return 100.0 * balanced_accuracy_score(y_true, y_pred)
        if metric == 'macro_f1':
            return 100.0 * f1_score(y_true, y_pred, average='macro', zero_division=0)
        if metric == 'mcc':
            # -1..1 mapped to 0..100 (ranking unchanged)
            return 100.0 * (matthews_corrcoef(y_true, y_pred) + 1.0) / 2.0
        if metric == 'gmean':
            rec = recall_score(y_true, y_pred, average=None, zero_division=0,
                               labels=sorted(set(y_true)))
            return 100.0 * float(np.prod(rec) ** (1.0 / len(rec))) if len(rec) else 0.0
        raise ValueError(
            "unknown fitness_metric=%r. Use accuracy, balanced_accuracy, macro_f1, mcc or gmean" % metric)

    def evaluate(self, df):
        """Grow the tree on the discretized training data and set the fitness.

        Returns the chefboost training accuracy (0-100)."""
        df_disc = discrete_df(df, self)
        self.config["feature_value_map"] = build_feature_value_map(df_disc)
        # per-process rule-file name so that several processes can run side by side
        self.model = tree.fit(df_disc, self.config, self.target, f'model_test_{os.getpid()}')

        accuracy = self.model['evaluation']['train']['Accuracy']
        self.fitness_value = self.fitness_from(self.model['evaluation']['train'])
        self.objective = self.fitness_value
        # release the prediction arrays; the chromosome keeps self.model alive
        _ev = self.model['evaluation']['train']
        _ev.pop('Actuals', None)
        _ev.pop('Predictions', None)
        return accuracy

    def save_model(self, df, model_name):
        """Refit on df and keep the rule file under ``model_name``."""
        df_disc = discrete_df(df, self)
        self.model = tree.fit(df_disc, self.config, self.target, f'{model_name}')

    def train_metrics_macro(self, df):
        """Macro P/R/F1 (0-100) of the current model on the training data."""
        df_disc = discrete_df(df, self)
        x_train = df_disc.drop(columns=[self.target])
        y_true = df_disc[self.target].astype(str)
        y_pred = [str(tree.predict(self.model, row.tolist())) for _, row in x_train.iterrows()]
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
        return precision, recall, f1


def discrete_df(df: pd.DataFrame, chromo: Chromosome):
    """Discretize the numerical attributes of df according to the chromosome."""
    df_copy = copy.deepcopy(df)
    for gene in chromo.genes:
        if gene.disc_num == 1:
            continue  # attribute stays continuous

        bin = [float('-inf')]
        label = []
        for i in range(len(gene.break_pts)):
            bin.append(gene.break_pts[i])
            label.append('Less_{0}'.format(gene.break_pts[i]))
        bin.append(float('inf'))
        label.append('Greater_{0}'.format(gene.break_pts[i]))

        gene.bin = bin
        gene.label = label
        df_copy[gene.feature] = pd.cut(df_copy[gene.feature], bins=bin, labels=label)
        df_copy[gene.feature] = df_copy[gene.feature].astype('object')
    return df_copy


def evolve(df_train, cat_cols, config):
    """Run the genetic search on one training fold.

    Each generation: the top ``survival_rate`` share of the population survives; the rest
    is refilled by mutation, crossover and fresh random chromosomes in the ratio
    mutation_rate : crossover_rate : random_rate. Parents are drawn uniformly from the
    survivors. Survivors are re-ranked together with the offspring every generation;
    identical genotypes are not refitted (fitness cache).

    Returns (best_chromosome, best_train_accuracy, best_fitness, n_fitness_calls, n_tree_fits).
    """
    pop_size = config['population_size']
    past_population = [
        Chromosome(list(df_train.columns), df_train, cat_names=cat_cols, target='Decision', config=config)
        for _ in range(pop_size)]

    mutation_size = int(pop_size * config['mutation_rate'])
    crossover_size = int(pop_size * config['crossover_rate'])
    survival_size = int(pop_size * config['survival_rate'])
    random_size = pop_size - mutation_size - crossover_size - survival_size
    if random_size < 0:
        raise ValueError("mutation/crossover/survival rates sum to > 1.0")

    fitness_cache = {} if config.get('fitness_cache', True) else None
    offspring_total = pop_size - survival_size
    n_fitness_calls = 0
    n_tree_fits = 0
    global_best = None
    global_best_metric = -1

    for generation in range(config['generations']):
        new_population = past_population.copy()
        n_off = offspring_total

        denom = 1.0 - config['survival_rate']
        mutation_size = int(n_off * config['mutation_rate'] / denom)
        crossover_size = int(n_off * config['crossover_rate'] / denom)
        random_size = n_off - mutation_size - crossover_size

        for _ in range(mutation_size):
            parent = random.choice(past_population)
            new_population.append(parent.mutate())

        for _ in range(crossover_size):
            parent1 = random.choice(past_population)
            parent2 = random.choice(past_population)
            new_population.append(parent1.crossover(parent2))

        for _ in range(random_size):
            new_population.append(
                Chromosome(list(df_train.columns), df_train, cat_names=cat_cols, target='Decision', config=config))

        evaluated_population = []
        for individual in new_population:
            n_fitness_calls += 1
            cache_key = individual.signature() if fitness_cache is not None else None
            cached = fitness_cache.get(cache_key) if cache_key is not None else None
            if cached is not None:
                accuracy, fitness_value = cached
                individual.fitness_value = fitness_value
                individual.objective = fitness_value
            else:
                accuracy = individual.evaluate(df_train)
                n_tree_fits += 1
                if cache_key is not None:
                    fitness_cache[cache_key] = (accuracy, individual.fitness_value)
            evaluated_population.append([individual, accuracy])

        # stable sort: on ties, earlier entries (survivors) stay ahead
        evaluated_population.sort(key=lambda x: x[0].fitness_value, reverse=True)

        past_population = [individual for individual, _ in evaluated_population[:survival_size]]

        best = evaluated_population[0]
        best_metric = best[0].fitness_value
        if best_metric > global_best_metric:
            global_best = best
            global_best_metric = best_metric

    best_individual, best_accuracy = global_best
    return best_individual, best_accuracy, global_best_metric, n_fitness_calls, n_tree_fits
