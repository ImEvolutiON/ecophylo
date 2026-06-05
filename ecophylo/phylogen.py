# -*- coding: utf-8 -*-
"""
Created on Fri Nov 6 13:20:00 2020

@author : Maxime Jaunatre <maxime.jaunatre@yahoo.fr>
@author : Elizabeth Bathelemy <barthelemy.elizabeth@gmail.com>

Functions : 
    toPhylo
    ubranch_mutation

"""
# TODO : more info in help toPhylo

import numpy as np

def toPhylo(tree, mu, tau = 0, spmodel = "loose", 
            force_ultrametric = True, seed = None):
    """
    Merge branches of genealogy following speciation model of the user choice 
    after sprinkling mutation events over the branches of simulated genealogies
    depending on branch lengths. 
    
    Mutation events are sprinkled over the branches of simulated genealogies 
    depending on branch lengths, so that the number of mutations over a branch 
    follows a Poisson distribution with parameter 𝜇·𝐵 where 𝜇 is the point 
    mutation rate and 𝐵 is the length of the branch. 
    
    The descendants stemming from a branch with at least one mutation define
    a genetically distinct clade. Since an extant species should be a 
    monophyletic genetic clade distinct from other species, all paraphyletic 
    clades of haplotypes at present are merged to form a single species. 
    
    Parameters
    ----------
    tree : TreeNode (ete3 class)
        A tree representing the genealogy of simulated individuals.
    mu : float
        point mutation rate, must be comprised between between 0 and 1. 
    tau = 0 : float
        The minimum number of generations monophyletic lineages have to be 
        seperated for to be considered distinct species
    spmodel = "SGD" : string
        the type of speciation model to implement. Default if "SGD" and 
        corresponds to a generalisation of the Speciation by Genetic 
        Differentiation. Note that setting tau to 1 will equate to the SGD 
        model as described in Manceau et al. 2015.
        "NTB" corresponds to the speciation model as described in Hubbell 2001, 
        in which point mutations instantenously give rise to new species.
    force_ultrametric = True : bool
        Whether or note to force phylogenetic tree ultrametry 
    seed = None : int
        None by default, set the seed for mutation random events.

    Returns
    -------
    Tree Node (ete3 class)
        A phylogeny representing the phylogenetic relationships among species 
        as well as the number of individuals descending from a speciation event
        in the genealogy, which defined the species abundance in the sample at 
        present (abundances can be retrived using the getAbund function).

    Examples
    --------
    >>> from ete3 import Tree
    >>> tree = Tree('(((A:5,(B:3, C:3))1:2,(D:2, E:2)1:5)1:2, (F:3, G:3)1:6);')
    >>> print(tree)
    <BLANKLINE>
             /-A
          /-|
         |  |   /-B
         |   \-|
       /-|      \-C
      |  |
      |  |   /-D
    --|   \-|
      |      \-E
      |
      |   /-F
       \-|
          \-G
    >>> phylo = toPhylo(tree, 0.5, seed = 42)
    >>> print(phylo)
    <BLANKLINE>
          /-sp1
       /-|
    --|   \-sp2
      |
       \-sp3
    >>> import ecophylo as eco
    >>> eco.getAbund(phylo, 7)
    [3, 2, 2]
    >>> phylo = toPhylo(tree, mu = 0.5, tau = 0.005, seed = 42)
    >>> print(phylo)
    <BLANKLINE>
       /-sp1
    --|
       \-sp2
    >>> phylo = toPhylo(tree, mu = 0.5, spmodel = "NTB", tau = 0.005, seed = 42)
    >>> print(phylo)
    <BLANKLINE>
       /-sp1
    --|
       \-sp2
    """
    # Idiot proof
    if tree.__class__.__name__ != 'TreeNode' :
        raise ValueError('tree must have a class TreeNode')
    if mu < 0 or mu > 1 or not isinstance(mu, (int,float)):
        raise ValueError('mu must be a float between 0 and 1')
    if not spmodel in ['loose', 'lacy', 'genealogy', 'broken-NTB']:
        raise ValueError(spmodel+' is not a correct model. '+
                'spmodel must be either "loose" or "lacy" string')
    if not isinstance(force_ultrametric, bool):
        raise ValueError('force_ultrametric must be a boolean')
    if seed is not None and not isinstance(seed, int):
        raise ValueError('seed must be an integer')
    if seed is not None:
        np.random.seed(seed) # Initialize RNG vector

    # init some parameters
    innerNodeIndex = 0
    nIndsORI = 0
    spID = 1
    demeID = 0
    ndeme = 0
    
    # mutation model on branches
    for node in tree.traverse("preorder"): # traverse les noeuds
        try:
            node.sp
        except AttributeError:
            node.add_features(sp=1)
        try:
            node.mut
        except AttributeError:
            node.add_features(mut="")
        try:
            node.deme
        except AttributeError:
            node.add_features(deme=1)

        if not node.is_leaf():
            node.name = "n%d" % innerNodeIndex
            innerNodeIndex += 1
        else:
            nIndsORI += 1
            name_deme = node.name.split("_")
            if len(name_deme) == 1: # if no population added, only one deme
                name_deme.append('0')
            if(int(name_deme[1]) > ndeme):
                ndeme += 1
            node.deme = int(name_deme[1])
            node.name = name_deme[0]

        if not node.is_leaf():
            umut = ubranch_mutation(node= node, mu= mu, tau= tau)
            if umut:
                # print(f"Speciation event @ node {node.name}")
                spID += 1
                node.sp = spID
                node.mut = "*"
                for leaf in node:
                    try:
                        leaf.sp = spID
                    except AttributeError:
                        leaf.add_features(sp=1)
            # print(f"node {innerNodeIndex} --> sp: {node.sp}")
        else :
            umut = ubranch_mutation(node= node, mu= mu, tau= tau)
            if umut :
                spID +=1
                node.sp = spID
                node.mut = "*"
    
    from collections import Counter
    
    leaf_names = []
    for leaf in tree.iter_leaves():
        leaf_names.append(leaf.sp)
    res = len(Counter(leaf_names).keys())
    #print(f'spCount = {res}')
    
    
    #===================================
    # PHYLOGENY WITH PARAPHYLETIC GROUPS
    #===================================
    #
    # it is necessary to find a way to represent the genealogy as a phylogeny
    # else, LTT and any phylogenetic analysis would be biaised.
    # As for now, integration of the UNTB is done only for present pattern
    # emerging from past demography; there's a Counter in "sumstat" that gets
    # the abundances per species : getAbund returns a correct sfs as
    # paraphyly don't have impact on present patterns of species when you
    # specify spmodel = "NTB". At least it shouldn't (TBD).
    #
    # Still, what should be done for the next patch :
    # - dev a new phylogenetic algorithm
    # - adapt the sumstat.py, and try to remove the needed "spmodel"
    # - remove this block
    #
    if spmodel == "broken-NTB":
        nsp = 1
        for leaf in tree.iter_leaves():
            popInd = [0] * (ndeme+1)
            popInd[leaf.deme] = 1
            leaf.popInd = popInd
            leaf.name = "sp"+str(nsp)
            nsp += 1
    
    if spmodel == "genealogy":
        for leaf in tree.iter_leaves():
            popInd = [0] * (ndeme+1)
            popInd[leaf.deme] = 1
            leaf.popInd = popInd

    if spmodel == "loose" : 

        traversed_nodes = set()
        
        for leaf in tree.iter_leaves():
            popInd = [0] * (ndeme+1)
            popInd[leaf.deme] = 1
            leaf.popInd = popInd
            
            # Here we built a binary matrice indicating in which deme the leaf is present and absent
            
        for node in tree.traverse("preorder"):
            if not node.is_leaf() and node not in traversed_nodes:
                # print(tree.get_ascii(attributes=["mut", "name"], show_internal=True))
                # print("\n" + "="*80)
                # print(f"\033[1;32mCURRENT NODE: {node.name}\033[0m")
                # print("="*80)
                children = node.get_children()
                if len(children) != 2:
                    raise ValueError("The algorithm does not know how to deal with non dichotomic trees.")
                left_species = {leaf.sp for leaf in children[0].iter_leaves()}
                right_species = {leaf.sp for leaf in children[1].iter_leaves()}
                # print(f"children = {[c.name for c in children]}")
                # print(f"left_species = {left_species}")
                # print(f"right_species = {right_species}")
                if not (left_species & right_species):
                    continue
            
                # Non-monophyletic group detected
                
                parent = node.up
                collapse_node = node
                traversed_nodes.update(collapse_node.traverse())
                
                subtree_leaves = list(collapse_node.iter_leaves())
                merged_ind = " ".join(leaf.name for leaf in subtree_leaves) # a string containing all merged individuals
                popInd = [0]*(ndeme+1) # a string containing all merged individuals but separated in demes
                for leaf in subtree_leaves:
                    popInd[leaf.deme] += 1
                
                new_leaf, new_leaf_dist = collapse_node.get_farthest_leaf()
                new_dist = new_leaf_dist + (collapse_node.dist if parent is not None else 0) # if parent is the root it's None, so dist = 0

    
                replacement_node = type(tree)()

                replacement_node.name = new_leaf.name
                replacement_node.dist = new_dist
                replacement_node.popInd = popInd
                replacement_node.mergedInd = merged_ind
                replacement_node.sp = new_leaf.sp
                
                collapse_node.detach()
                
                if parent is None:
                    for child in list(tree.get_children()):
                        child.detach()
                
                    tree.add_child(replacement_node)      
                else:
                    parent.add_child(replacement_node)
            # else:
            #     print(tree.get_ascii(attributes=["mut", "name"], show_internal=True))
            #     print("END OF PREORDER")
                    
                # There's a float miscalculation that breaks the ultrametricity of an order of 1 float unity (~1e-16)
                # It should be invisible to most softwares. But still. The "force_ultrametricity" part of the code
                # doesn't work but it can be improved and maybe will fix the ultrametricity ? I put a check just before
                # return tree that check if ultrametricity is broken at a bigger threshold (1e-12) it will only print in the
                # logs so be careful !
                
    
    if spmodel == "lacy" :
        
        traversed_nodes = set()
        
        for leaf in tree.iter_leaves():
            popInd = [0] * (ndeme+1)
            popInd[leaf.deme] = 1
            leaf.popInd = popInd
            
        for node in tree.traverse("postorder"):
            if not node.is_leaf() and node not in traversed_nodes:
                # print(tree.get_ascii(attributes=["mut", "name"], show_internal=True))
                # print("\n" + "="*80)
                # print(f"\033[1;32mCURRENT NODE: {node.name}\033[0m")
                # print("="*80)
                children = node.get_children()
                if len(children) != 2:
                    raise ValueError("The algorithm does not know how to deal with non dichotomic trees.")
                left_species = {leaf.sp for leaf in children[0].iter_leaves()} # it may happen that node is an inner node, and children are inner nodes too. Neither of them have sp attribute
                right_species = {leaf.sp for leaf in children[1].iter_leaves()} # we then need to check the sp attribute of leaves attached to children.
                if left_species != right_species :
                    continue
                
                # Monophyletic detected ; either this is the first iteration so children are true leaves, eitheir this is n-th iteration and
                # children were collapsed (now called false-leaves) and already have popInd and mergedInd values that we need to compute and
                # not simply increment as this was the case in preorder for the "loose" definition
                
                parent = node.up
                collapse_node = node
                traversed_nodes.update(collapse_node.traverse())
                
                replacement_node = type(tree)()
                
                new_leaf, new_leaf_dist = collapse_node.get_farthest_leaf()
                new_dist = new_leaf_dist + (collapse_node.dist if parent is not None else 0) # if parent is the root it's None, so dist = 0
                    
                merged_ind = " ".join(child.mergedInd if hasattr(child, "mergedInd") else child.name for child in collapse_node.get_children())
                popInd = [0]*(ndeme+1)
                for leaf in collapse_node.get_children(): 
                    for i in range(ndeme+1):
                        popInd[i] += leaf.popInd[i] # sum of both leaves
                
                replacement_node.name = new_leaf.name
                replacement_node.dist = new_dist
                replacement_node.popInd = popInd
                replacement_node.mergedInd = merged_ind
                replacement_node.sp = new_leaf.sp

                collapse_node.detach()
                
                if parent is None:
                    for child in list(tree.get_children()):
                        child.detach()
                
                    tree.add_child(replacement_node)      
                else:
                    parent.add_child(replacement_node)
                    
                                    
                
    # if spmodel == "phenotypic" :
        
    # if spmodel == "paraphyletic" :
        
    # if spmodel == "genealogy" :
        
    
    if force_ultrametric: # TODO : add is.ultramtric from ete3, maybe this will work for float loss
         tree_dist = tree.get_farthest_leaf()[1]
         for leaf in tree.iter_leaves():
             dst = tree.get_distance(leaf)
             if dst != tree_dist:
                 leaf.dist += tree_dist - dst


    if spmodel in ("loose", "lacy"):
        nsp = 1
        for leaf in tree.iter_leaves():
            leaf.name = "sp"+str(nsp)
            nsp += 1
            
            
    # SECURITY CHECK FOR ULTRAMETRICITY FLOAT LOSS
            
    ages = [tree.get_distance(leaf) for leaf in tree.iter_leaves()]
    delta_abs = max(ages) - min(ages)
    delta_rel = delta_abs / max(ages) if max(ages) else 0.0
    
    if delta_rel > 1e-12:
        print("\n==============================================================================================")
        print("ULTRAMETRICITY BROKEN AT A 1e-12 ORDER (float error should broke at a 1e-16 order, that's weird)")
        print("================================================================================================")
        print("tips:", len(ages), " species")
        print("min:", min(ages))
        print("max:", max(ages))
        print("delta abs:", delta_abs)
        print("delta rel:", delta_rel)
    
    
    return tree


def ubranch_mutation(node, mu, tau = 0, seed = None):
    """
    Draw mutations following a poisson process with parameter 
    max((B - tau), 0)*mu where mu is the point mutation rate, B is the 
    length of the branch at a given node and tau 

    Parameters
    ----------
    node : ete3.coretype.tree.TreeNode
        node from which to compute branch length
    mu : float
        point mutation rate, must be comprised between between 0 and 1. 
    tau = 1 : float
        The minimum number of generations monophyletic lineages have to be 
        seperated for to be considered distinct species
    seed : int
        None by default, set the seed for mutation random events.
        ## FIX with Dependent-patch ##
        When calling a global simulation pipeline with ecophylo.simulate()
        genealogy to phylogeny the RNG vector is called at the beginning
        of toPhylo() so that it's not reset each time ubranch_mutation() is
        called. If you want to check on the behaviour of ubranch_mutation() 
        with seeding, you can set it and it will reset the RNG vector each
        time ubran_mutation() is run.
        
    Returns
    -------
    bool
        whether or not a at least one mutation should appear on the tree at 
        this node

    Examples
    --------
    TODO: Examples with tau ?
    >>> from ete3 import Tree
    >>> tree = Tree('((A:1,(B:1,C:1)1:1)1:5,(D:1,E:1)1:1);')
    >>> node = tree.children[0] # first non-root node
    
    >>> ubranch_mutation(node = node, mu = 0, seed = 42)
    False
    
    >>> ubranch_mutation(node = node, mu = 1, seed = 42)
    True
    
    >>> ubranch_mutation(node = node, mu = 0.5, seed = 42)
    True
    """
    # Idiot proof
    if node.__class__.__name__ != 'TreeNode' :
        raise ValueError('node must have a class TreeNode')
    if mu < 0 or mu > 1 or not isinstance(mu, (int,float)):
        raise ValueError('mu must be a float between 0 and 1')
    if tau < 0 or not isinstance(tau, (int,float)):
        raise ValueError('tau must be a float superior or equal to 0')
    if seed is not None and not isinstance(seed, int):
        raise ValueError('seed must be an integer')
    if seed is not None:
        np.random.seed(seed) # if you put a seed in umut_branch() it means you want to check its behavior so reset seed
    
    lambd = max((node.dist - tau), 0) * mu 
    rb = np.random.poisson(lambd) 
    return rb >= 1 # parametrize the 1 by a value n


if __name__ == "__main__":
        import doctest
        doctest.testmod()
