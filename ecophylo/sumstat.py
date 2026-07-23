# -*- coding: utf-8 -*-
"""

Created on Wed May 13 11:42:50 2020

@author : Maxime Jaunatre <maxime.jaunatre@yahoo.fr>
@author : Elizabeth Bathelemy <barthelemy.elizabeth@gmail.com>
@author : Théo Driancourt <imevolutionmodding@gmail.com>

Functions :
    getLTT
    getAbund
    getDeme

"""

import numpy as np

def getLTT(tree, spmodel = None, age = "mutation", graph = False):
    """
    Description
    ----------
    Compute a Label/Lineage-Through-Time curve.
    
    Parameters
    ----------
    tree : (ete3 class)
        Phylogeny or genealogy with attribute "spmodel" on the root. Output of
        toPhylo().
    
    spmodel : {"genealogy", "loose", "lacy", "phenotypic"}, string, red
    automatically
        The species partition used to transform the gene tree in a species
        tree.
        
        - "genealogy"
        Compute a genealogies-through-time curve from the topology of the
        individual genealogy.
        
        - "loose", "lacy"
        Compute a species lineage-through-time curve from the phylogenetic nodes
        of the monophyletic species tree returned by the species partition in
        toPhylo().
        
        - "phenotypic"
        Compute a label-through-time curve from a mutation table keeping the
        date of appearance of mutations a label, the nearest and the oldest
        coalescence time of this label and its parent label. LTT produced are
        modulated by the "age" argument.
    
    age : {"mutation", "nearest", "oldest"}, string, default = "mutation"
        MRCA convention for non-monophyletic label histories. Ignored when a
        monophyletic partition is provided {"loose", "lacy"} and when working 
        on a genealogy. Lambert Fig3,4.
        
        - "mutation"
        Use the mutation / derived-character appearance time to define the
        divergence time. Mostly used in UNTB approaches (Jabot & Chave, ...)
        
        - "nearest"
        Use the most recent coalescence time between all pairs of individuals 
        carrying the derived label and all individuals carrying the parent
        label to define divergence time of the new label-group.
        
        - "oldest"
        Use the oldest coalescence time between all pairs of individuals 
        carrying the derived label and all individuals carrying the parent
        label to define divergence time of the new label-group.
        
        
    graph : bool, default = False
        If True, plot the LTT curve with plotnine in output["plot"]
        
    Output
    ----------
    dict : Dictionnary with
            - "time" : time from the oldest retained event to present
            - "age" : age before present
            - "leneages" : number of lineages
            - "spmodel" : species partition
            - "age_type" : age convention used
            - "plot" : plotnine object, only if graph = True
    """
    if tree.__class__.__name__ != "TreeNode":
        raise ValueError('Tree must have class TreeNode')
    if spmodel is None:
        if hasattr(tree, "spmodel"):
            spmodel = tree.spmodel
        else:
            raise ValueError("spmodel must be provided because the tree has no 'tree.spmodel' attribute."
                             "Maybe you didn't use toPhylo() ?")
    if spmodel not in ("genealogy", "loose", "lacy", "phenotypic") :
        raise ValueError("spmodel must be either 'genealogy', 'loose', 'lacy' or 'phenotypic'.")
    if spmodel == "genealogy":
        raise ValueError("spmodel = genealogy : A genealogy cannot be computed into LTT yet.")
    if spmodel not in ("loose", "lacy", "phenotypic"):
        raise ValueError('spmodel must be one of "genealogy", "loose", "lacy" or "phenotypic".')
    if spmodel == "phenotypic" and age not in ("mutation", "nearest", "oldest"):
            raise ValueError('age must be one of "mutation", "nearest" or "oldest".')
    if not isinstance(graph, bool):
        raise ValueError('graph must be a boolean.')
    leaves = list(tree.iter_leaves())
    if len(leaves) == 0:
        raise ValueError('tree has no leaves.')
    leaf_depths = [tree.get_distance(leaf) for leaf in leaves]
    tree_height = max(leaf_depths)
    delta_abs = max(leaf_depths) - min(leaf_depths)
    delta_rel = delta_abs / max(leaf_depths) if max(leaf_depths) else 0.0
    
    if delta_rel > 1e-8 :
        print("/!\\ WARNING /!\\ tree is not ultrametric at tolerance 1e-8. Ages are computed using the farthest leaf.")


    # Event collector
    events = {} # Dictionary storing date of events
    
    if spmodel in ["loose", "lacy"] :
        print("here iz ur ltt : a lot of lineages :3")
    elif spmodel == "phenotypic" :
        # Label-Through-Time curve
      
        start_lineages = 0 # We start with 0 retained labels
        age_type = age
        
        try :
            mutation_table = tree.mutation_table
        except AttributeError :
            raise ValueError("getLTT(spmodel = phenotypic) requires tree.mutation_table."
                             'If you generated a genealogy with simulate(spmodel = "genealogy") you should use toPhylo() before getLTT().'
                             'However if you generated a phylogeny with simulate then something has gone wrong.')
        
        age_column = {"mutation": "mutation_coalescence_age",  # Translate the user argument into the mutation_table column name. 
                      "nearest": "nearest_coalescence_age",    # If age is "mutation" the column name isn't "mutation" its
                      "oldest": "oldest_coalescence_age"}[age] # "mutation_coalescence_age"
            
        events[tree_height] = events.get(tree_height, 0) + 1 # Count ancestral label at tree root. If it's not already assigned set it to 0 ; then add 1

        for row in mutation_table :
            if age_column not in row :
                raise ValueError('Column '  + age_column + 'is missing from mutation_table. Check toPhylo()')
            
            label_age = row[age_column] # read the row at column "age" as defined and translated from user
            
            if label_age is None : # because we initialized all oldest/nearest_coalescence_age to None, check if there's a None remaining
                raise ValueError(
                    'Cannot compute phenotypic getLTT(age = "'+ age +'") because '
                    'mutation ' + str(row.get("mutation_id")) + ' has no value for '
                    + age_column)
            
            label_age = float(label_age)
            
            if np.isnan(label_age) :
                raise ValueError("Cannot compute phenotypic getLTT(age='" + age + "')because "
                                 "mutation " + str(row.get("mutation_id")) + " has NaN for " +
                                 age_column + ".")
            if label_age < 0 :
                raise ValueError('Label age cannot be negative.')
                
            if label_age > tree_height + 1e-8 :
                raise ValueError('Label age cannot be older than tree height.')
                
            if abs(label_age) < 1e-12 : # Clean tiny float values close to zero, should never happen but we never know
                label_age = 0.0
                print('Label happened too close to present, at a rate smaller than 1e-12 coalescent time, set it to present (0)')
            
            events[label_age] = events.get(label_age, 0) + 1 # if events[label_age] doesn't exist yet, get() returns 0
                                                             # we add 1 to it and assign it back.
                                                             # if events[label_age] already exists, get() returns the associated value of the key
                                                             # we add 1 to it and assign it back.
            
    # Build LTT
        
    if len(events) == 0 :
        if spmodel == "phenotypic":
            raise ValueError('No event found for getLTT(spmodel="phenotypic"). This shouldn\'t happen'
                             ' because getLTT() explicity adds ancestral label 1 at tree_height.')
            # This is possible for single tip loose/lacy trees
            
        output_age = [tree_height]
        output_lineages = [1]
        if not np.isclose(tree_height, 0.0, atol = 1e-12): # VERIFIER IMPLEMENTATION DE SINGLE TIP LOOSE/LACY
            output_age.append(0.0)
            output_lineages.append(1)
    else:
        sorted_ages = sorted(events.keys(), reverse=True) # Sort events from oldest to youngest
        output_age = []
        output_lineages = []
        current_lineages = start_lineages
        
        for event_age in sorted_ages:
            current_lineages += events[event_age]
            output_age.append(event_age)
            output_lineages.append(current_lineages)
        if not np.isclose(output_age[-1], 0.0, atol=1e-12): # Extend the curve to present if the last (youngest) isnt close to 0
            output_age.append(0.0)
            output_lineages.append(current_lineages)
    
    oldest_retained_event = output_age[0]
    output_time = [oldest_retained_event - x for x in output_age] # Convert age before present into time since oldest retained event
    
    # Output
    
    output = {"time" : output_time,
              "age" : output_age,
              "lineages" : output_lineages,
              "spmodel" : spmodel,
              "age_type" : age_type}
    
    if graph:
        try:
            import pandas as pd
            import plotnine as p9
        except ImportError:
            raise ImportError("graph = True requires pandas and plotnine to be installed on this machine.")
        
        df = pd.DataFrame({
            "time": output_time,
            "age": output_age,
            "lineages": output_lineages})
        
        output["plot"] = (
            p9.ggplot(df, p9.aes(x="time", y="lineages"))
            + p9.geom_step()
            + p9.theme_classic()
            + p9.labs(
                x="time from oldest retained event to present",
                y="Number of lineages / labels",
                title = "Through-Time Curve"))
    return output

            
def getAbund(tree, samples = None, spmodel = None):
    """
    Description
    ----------
    Compute species abundance from a species phylogeny having labels of merged
    individuals resulting from the merging process of toPhylo() - a function
    used to give the species tree of a gene tree.
    
    Parameters
    ----------
    tree : (ete3 class)
        Phylogeny with attributes on leaves. This attributes is a character 
        string containing all names of the species individual (mean to use 
        topPhylo result). Names is formated like this :
        " name1 name2 name3"
    
    spmodel : {"genealogy", "loose", "lacy", "phenotypic"} string
        The species partition used to transform the gene tree in a species tree
        that have impact on how to read labels. Genealogy and phenotypic can
        have paraphyletic groups to you have to add the individuals of two
        different tips.
    
    samples : int
        number of individual in the community
        # TODO : set this check as optionnal

    Output
    -------
    sfs : array of int
        A Species Frequency Spectrum of the sampled tree, size being the number
        of species, values being the abundance of each species.
        
        Note that you can np.sort(ecophylo.getAbund(tree))[::-1] to obtain the
        RAD, Rank Abundance Distribution.
        
    Examples
    --------
    >>>
    """
    # Idiot proof
    if tree.__class__.__name__ != 'TreeNode' :
        raise ValueError('tree must have a class TreeNode')
    if samples != None:
      if not isinstance(samples, int):
          raise ValueError('samples must be an integer')

    abund = list()
    sfs = list()
            
    if spmodel is None:
        if hasattr(tree, "spmodel"):
            spmodel = tree.spmodel
        else:
            raise ValueError("spmodel must be provided because the tree has no 'tree.spmodel' attribute")

    if spmodel not in ("loose", "lacy", "phenotypic", "genealogy"):
        raise ValueError("spmodel must be either 'loose', 'lacy', 'genealogy' or 'phenotypic'")

            
    if spmodel in ("loose", "lacy"):
        for leaf in tree.iter_leaves():
            try:
                inds = leaf.mergedInd.lstrip(" ") # remove 1st space
                inds = list(inds.split(" ")) # strip on spaces
                abund.append(len(inds)) # get length 
            except AttributeError:
                abund.append(1) #shouldn't this be a list so append([1]) ?
        sfs.extend(abund)
    elif spmodel in ("genealogy", "phenotypic"): #CHANGER ICI GENEALOGY MARCHE PAS
        from collections import Counter
        leaf_names = []
        for leaf in tree.iter_leaves():
            leaf_names.append(leaf.sp)
        sfs = list(Counter(leaf_names).values())
    else:
        raise Exception('spmodel should be loose or lacy')
        # think about catching error when phylogeny has only 1 sp
        
    if samples != None and sum(sfs) != samples:
        raise Exception("Simulated phylogeny has only one species!")
        # TODO : modify error with a better check here
    return sfs

def getDeme(tree, div = False, spmodel = None):
    """
    
    Parameters
    ----------
    tree : (ete3 class)
        Phylogeny with attributes on leafs. This attributes is a character 
        string containing all names of the species individual (mean to use 
        topPhylo result). Names is formated like this :
        " name1 name2 name3"
    div : bool
        Option to simplify the matrix to a simple list of Deme species diversity.

    Returns
    -------
    indiv : nest list of int
        site/species matrix of the tree

    Examples
    --------
    >>> from ete3 import Tree
    >>> tree = Tree('(((A_0:5,(B_0:3, C_1:3))1:2,(D_1:2, E_1:2)1:5)1:2, (F_2:3, G_2:3)1:6);')
    >>> import ecophylo as eco
    >>> phylo = eco.toPhylo(tree, 0.5, seed = 42)
    >>> print(phylo)
    <BLANKLINE>
          /-sp1
       /-|
    --|   \-sp2
      |
       \-sp3
    >>> getDeme(phylo)
    [[2, 1, 0], [0, 2, 0], [0, 0, 2]]
    >>> getDeme(phylo, div = True)
    [1, 2, 1]
    """
    # Idiot proof
    if tree.__class__.__name__ != 'TreeNode' :
        raise ValueError('tree must have a class TreeNode')
    
    indiv = list()
    
    if spmodel is None:
        if hasattr(tree, "spmodel"):
            spmodel = tree.spmodel
        else:
            raise ValueError("spmodel must be provided because the tree has no 'tree.spmodel' attribute")
            
    if spmodel in ("loose", "lacy"): 
        for leaf in tree.iter_leaves():
            try:
                indiv.append(leaf.popInd)
            except AttributeError :
                indiv.append([1])
    
    elif spmodel in ("genealogy", "broken-NTB"):
        sp_to_popInd = {}
        for leaf in tree.iter_leaves():
            sp = leaf.sp
            if sp not in sp_to_popInd:
                sp_to_popInd[sp] = leaf.popInd.copy()
            else:
                for i in range(len(leaf.popInd)):
                    sp_to_popInd[sp][i] += leaf.popInd[i]
        indiv = list(sp_to_popInd.values())
    
    else:
        raise ValueError("spmodel should be loose, lacy or genealogy")
    
    if div:
        indiv = np.array(indiv)
        indiv = [sum(indiv[:,i] > 0) for i in range(indiv.shape[1])]
        # TODO : t(indiv) and compute div 
    return indiv