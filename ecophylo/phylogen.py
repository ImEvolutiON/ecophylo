# -*- coding: utf-8 -*-
"""
Created on Fri Nov 6 13:20:00 2020

@author : Maxime Jaunatre <maxime.jaunatre@yahoo.fr>
@author : Elizabeth Bathelemy <barthelemy.elizabeth@gmail.com>
@author : Théo Driancourt <imevolutionmodding@gmail.com>

Functions : 
    toPhylo
    ubranch_mutation
    

References :
    -  Manceau, M., Lambert, A. The Species Problem from the Modeler’s Point of
      View. Bull Math Biol 81, 878–898 (2019).
      https://doi.org/10.1007/s11538-018-00536-2
      ↳ Noted Manceau & Lambert (2019).
    - Michael A. Bender and Martin Farach-Colton,
      "The LCA Problem Revisited", LATIN 2000, Lecture Notes in Computer
      Science 1776, pp. 88-94, Springer-Verlag, 2000.
      ↳ Noted Bender & Farach-Colton (2000).
    

"""
# TODO : more info in help toPhylo

import math
import numpy as np

def toPhylo(tree, mu, tau = 0, spmodel = "paraphyletic", 
            force_ultrametric = True, age = "mutation", seed = None, debug = False):
    """
    Transform a genealogy of individuals according to species model and
    definition.
    Mutations are sprinkled independently under the infinite allele assumption
    according to a Poisson process, each mutational event is passed down to the
    nodes below and can be overwriten by any other mutational event. We used
    the Poisson process of parameters 𝜇·𝐵 where 𝜇 is the point mutation rate
    and 𝐵 is the length of the branch. Rather than counting total events
    number on the branch, we sample the first mutation using an exponential
    distribution of parameter 𝜇 (Waiting Time T) and checked if it was on the
    branch (T ≤ 𝐵 : the first mutation occured on the branch) or outside
    (T ≥ 𝐵 : the first mutation couldn't occur on the branch), see more in
    ubranch_mutation() documentation.
    
    The descendants stemming from a branch with at least one mutation define
    a genetically distinct clade. These clades can be paraphyletic because of
    ancestral retention. Either the user choose to accept paraphyletic gene
    histories ('spmodel = paraphyletic'), or to impose monophyly. For every
    gene tree exist a species partition satisfying either A or B, along M.
     - (A) : Heterotypy between species.
     - (B) : Homotoypy within species.
     - (M) : Monophyly.
    These differences in species tree can be compared to the taxonomist
    opposition between splitter and lumper. See Manceau & Lambert (2019), they
    called the two combinations (AM) and (BM) 'loose' and 'lacy' species
    partition and (AB) 'phenotypic' that we chose to call 'paraphyletic'.
    
    Parameters
    ----------
    tree : TreeNode (ete3 class)
        A tree representing the genealogy of simulated individuals.
        
    mu : float
        Point-mutation rate, must be comprised between between 0 and 1.
        
    tau = 0 : float
        The minimum number of generations monophyletic lineages have to be 
        seperated for to be considered distinct species
        The minimum time on a branch that have to pass for mutation to occur.
        
        
    spmodel = {"genealogy", "loose", "lacy", "paraphyletic"}
        default = "paraphyletic" : string

        - "genealogy"
        The complete genealogy is retained after mutation sprinkling.
        Cannot be forced into a species tree or a LTT (later implementation
        may allow genealogical-LTT).
        
        - "loose"
        The complete genealogy is collapsed in a set of monophyletic clades
        according to the "loose species partition" from Manceau & Lambert
        (2019).
        The loose species partition aims to respect heterotypy between species,
        individuals in different species are genetically different for each
        species cluster. This is the finest partition of the present-day
        individuals as it usually needs to merge different clades : individuals
        can have different labels within a species. This is equivalent to
        lumpers.
        
        - "lacy"
        The complete genealogy is collapsed in a set of monophyletic clades
        according to the "lacy species partition" from Manceau & Lambert
        (2019).
        The lacy species partition aims to respect homotypy within species,
        individuals within the same species are genetically identical for each
        species cluster. This is the coarsest partition of the present-day
        individuals as it usually needs to seperate two paraphyletic parts :
        individuals with the same labels can be in different species.
        This is equivalent to splitters.
        
        - "paraphyletic"
        The complete genealogy is collapsed in a set of monophyletic lineages
        as a "label-tree" rather than a species tree. The label-tree tries to
        represent relations between labels at the cost of the genealogy
        representativity and may produce unresolved nodes because of age
        conventions.
        
    age = {"mutation", "nearest", "oldest"}, default="mutation"
        Divergence-age convention for nodes used for spmodel="paraphyletic".
        When inspecting relations between species u and v, u being
        paraphyletic, then u and v can have multiple common ancestors on
        different nodes.

                              ┌──────*purple──── purple_ind1
                       ┌──────┤n2
                       │      └───────────────── grey_ind1
                       │
                       │
                       │                  ┌───── red_ind1
               grey  ──┤n1           ┌*red┤n6
                       │             │    └───── red_ind2
                       │      ┌─*blue┤n4
                       │      │      │        ┌─ yellow_ind1
                       │      │      └─*yellow┤n7
                       │      │               └─ yellow_ind2
                       └──────┤n3
                              │         ┌*green─ green_ind1
                              │      ┌──┤n8
                              │      │  └─────── grey_ind2
                              └──────┤n5
                                     │  ┌─────── grey_ind3
                                     └──┤n9
                                        └─────── grey_ind4

                      100     75    50    25    0 BP
    
        We can see here that all individuals of two species don't share the
        same common ancestor : in the relation red and grey, red_ind1 and
        grey_ind2 have a common ancestor at node n3. But red_ind1 and grey_ind1
        have a common ancestor at node n1. Manceau & Lambert (2019) pose 3
        conventions,
            1. to use a nearest age convention
            2. to use a oldest age convention.
            3. to use the timing of the mutation event as divergence node 
               (usually used in UNTB models).
        For "nearest" and "oldest", the definition found in Manceau &
        Lambert (2019) informs us "The first two possibilities consist in
        relying on a time of divergence between individuals of the newly
        derived species and individuals of the ancestral, mother, species".
        So to find this node, we have to compute distances pairwise for every
        individuals of each species u and v.
        Historical labels may also exist : DEFINITION TO BE DONE.
        In this list of ages, min() is the nearest phylogenetic node and max()
        is the oldest phylogenetic node.
        In this example,
        mutation :


                            ┌──────────── purple
                      ┌─────┤
                      │     │      ┌───── grey
                      │     └──────┤
                      │            └───── green
                      │    
                      ┤
                      │          ┌──────── red
                      └──────────┤
                                 └──────── yellow

                      70    50    25     0 BP
                      
        nearest :

                            ┌──────────── purple
                            │
                            │      ┌───── grey
                      ──────┼──────┤
                            │      └───── green
                            │
                            │      ┌───── red
                            └──────┤
                                   └───── yellow
                                   
                           50     25    0 BP
        oldest :

                            ┌──────────── grey
                            │
                            ├──────────── purple
                      ──────┤
                            ├──────────── green
                            │
                            │      ┌───── red
                            └──────┤
                                   └───── yellow
                                   
                    75     50     25    0 BP
        
        - "mutation"
        Use the age of apparition of the derived label on the genealogy to
        compute phylogenetic nodes.
        
        - "nearest"
        Use the shortest pairwise coalescence age between present-day
        representatives of the derived-side and the present-day representatives
        of the parent-side.
        
        - "oldest"
        Use the longest pairwise coalescence age between present-day
        representatives of the derived-side and the present-day representatives
        of the parent-side.
        
        However pairwise queries can take way too much computing time so it is
        implemented as an ±1-RMQ query (explanation within the code and this
        reference : Bender, Farach-Colton 2000).

    force_ultrametric = True : bool
        Whether or note to force phylogenetic tree ultrametry 
    seed = None : int
        None by default, set the seed for mutation random events.
    debug = False : bool
        Change the return function, returns a tuple containing the phylogeny,
        the number of species before the partitioning, the number of species
        after the partitioning, the number of singleton before the partitioning
        the number of singleton after the partitioning, the tmrca of the tree
        before the partitioning, the tmrca after the partitioning

    Output
    -------
    Tree Node (ete3 class)
        A phylogeny representing the phylogenetic relationships among species 
        as well as the number of individuals descending from a speciation event
        in the genealogy, which defined the species abundance in the sample at 
        present (abundances can be retrived using the getAbund function).
    
    Examples
    --------
    >>>
    """
    # User inputs
    if tree.__class__.__name__ != 'TreeNode' :
        raise ValueError('tree must have a class TreeNode')
    if mu < 0 or mu > 1 or not isinstance(mu, (int,float)):
        raise ValueError('mu must be a float between 0 and 1')
    if not spmodel in ['loose', 'lacy', 'genealogy', 'paraphyletic']:
        raise ValueError(spmodel+' is not a correct model. '+
                'spmodel must be either "loose", "lacy", "genealogy" or ',
                '"paraphyletic" string')
    if not isinstance(force_ultrametric, bool):
        raise ValueError('force_ultrametric must be a boolean')
    if not isinstance(debug, bool):
        raise ValueError('debug must be a boolean')
    if seed is not None and not isinstance(seed, int):
        raise ValueError('seed must be an integer')
    if seed is not None:
        np.random.seed(seed) # Initialize RNG vector
    if age not in ["mutation", "nearest", "oldest"]:
        raise ValueError('age must be either "mutation", "nearest" or ',
                         '"oldest"')

    #==========================================================================
    # MUTATION MOTOR ON THE GENEALLOGY
    #==========================================================================
    
    innerNodeIndex = 0
    nIndsORI = 0
    spID = 1
    demeID = 0
    ndeme = 0
    
    sp_origin = {1: tree} # Genealogy node where each label appears, 1 appears at the root
    tree_height = max(tree.get_distance(leaf) for leaf in tree.iter_leaves())
    mutation_table = [] # Records all mutation events
    
    # Assign mutations along the genealogy
    # 
    # The input genealogy is traversed from ancestors toward descendants
    # ("preorder"). Every node and leaf inherit its parent's label as when a
    # mutation occurs on a branch, it runs down on descendants.
    #                      ┌───── blue_ind1
    #                 ┌────┤
    #                 │    └*red─ red_ind1
    #     ───────*blue┤
    #                 │        ┌─ yellow_ind1
    #                 └─*yellow┤
    #                          └─ yellow_ind2
    # Here the mutation blue got overwrite by mutations red and yellow.
    
    for node in tree.traverse("preorder"):
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
            
        is_leaf = node.is_leaf() # check if current node is a leaf

        if node.up is None: # if we are at the root
            node.add_features(sp=1) # then set it as sp=1
            continue # no mutation can happen above a root : there's no branch
        else:
            node.add_features(sp=node.up.sp) # else inherit parent label
        
        umut, mut_from_parent = ubranch_mutation(node = node, mu = mu, tau = tau)
        # stochastic mutation event : see doc or code below of "ubranch_mutation"
        # to get more information.
        
        if not umut:
            continue
        
        # umut is True : a mutation occured on the branch between node and node.up
        # because we are working in an infinite allele assumption we can simply add
        # "1" to a global species naming variable spID.
        
        parent_label = node.sp # get parent sp from node before overwritting it
        spID += 1 # infinite allele global species naming variable
        node.sp = spID # overwrite the sp attribute
        sp_origin[spID] = node # remember that spID occured at node "node"
        node.mut = "*" # for debugging when printing the tree
        
        # For that mutation event, we record it in a mutation_table.
        
        parent_name = node.up.name
        child_name = node.name
        
        parent_depth = tree.get_distance(node.up)  
        mutation_depth = parent_depth + mut_from_parent
       
        mutation_coalescence_age = tree_height - mutation_depth
        mutation_id = str(parent_name) + "->" + str(child_name)
        
        node.add_features(mutation_id = mutation_id,
                          mut_from_parent = mut_from_parent,
                          mutation_coalescence_age = mutation_coalescence_age)
        
        mutation_table.append({
            "mutation_id": mutation_id,
            "is_terminal": is_leaf,
            "label": spID,
            "parent_label": parent_label,
            "mutation_coalescence_age": mutation_coalescence_age, # Lambert scenario c
            "nearest_coalescence_age": None, # Lambert scenario a
            "oldest_coalescence_age": None # Lambert scenario b
            })
    #================================================================================================================
    # "nearest" and "oldest" in paraphyletic trees
    #================================================================================================================
    # In a strictly monophyletic species tree, the divergence time between two species is defined by a single node
    # because each species is a clade. Therefore, the monophyletic node is also the divergence node.
    # This is no longer true when species are non-monophyletic, which is the case of the gene trees that we
    # want to transform into species trees. This multiple divergence node problem is due to ancestral retention that may
    # distribute a label across different part of the genealogy (see grey label in the genealogy below).
    #
    # Therefore, in a paraphyletic partition of species, different pairs of individuals belonging
    # to the same two species can have different MRCA ages (see grey -> green transition below).
    #
    # This is highlighted by Manceau & Lambert (2019), they propose three age conventions to assign a divergence node :
    #   nearest : the shortest coalescence time among all pairs of individuals from both species
    #   oldest : the longest coalescence time
    #   mutation : the date of birth of the derived character
    # However, the first two can produce polytomic topology. And the last one can produce phylogenetic
    # topology inconsistent with genealogical topology.
    #
    # Here's an exemple : 
    #                         ┌──────*purple──── purple_ind1
    #                  ┌──────┤n2
    #                  │      └───────────────── grey_ind1
    #                  │           
    #                  │      
    #                  │                  ┌───── red_ind1
    #          grey  ──┤n1           ┌*red┤n6
    #                  │             │    └───── red_ind2
    #                  │      ┌─*blue┤n4
    #                  │      │      │        ┌─ yellow_ind1
    #                  │      │      └─*yellow┤n7
    #                  │      │               └─ yellow_ind2
    #                  └──────┤n3
    #                         │         ┌*green─ green_ind1
    #                         │      ┌──┤n8
    #                         │      │  └─────── grey_ind2
    #                         └──────┤n5
    #                                │  ┌─────── grey_ind3
    #                                └──┤n9
    #                                   └─────── grey_ind4
    #
    #                 100     75    50    25    0 BP
    #
    # This genealogy gives us this mutation_table : 
    #       grey -> blue :
    #           - mutation = 70
    #           - nearest = 75 on n3 (special case)
    #           - oldest = 100 on n1 (special case)
    #       grey -> purple :
    #           - mutation = 50
    #           - nearest = 75 on n2 (because the shortest coalescence time in pairwise individuals is LCA(purple_ind1, grey_ind1))
    #           - oldest = 100 on n1 (because the longest coalescence time in pairwise individuals is LCA(purple_ind1, grey_ind2/3/4))
    #       blue -> red :
    #           - mutation = 45
    #           - nearest = 50 on n4
    #           - oldest = 50 on n4
    #       blue -> yellow :
    #           - mutation = 40
    #           - nearest = 50 on n4
    #           - oldest = 50 on n4
    #       grey -> green :
    #           - mutation = 27
    #           - nearest = 30 on n8 (because the shortest coalescence time in pairwise individuals is LCA(green_ind1, grey_ind2))
    #           - oldest = 100 on n1 (because the longest coalescence time in pairwise individuals is LCA(green_ind1, grey_ind1))
    #
    # Now in our model : 
    # For each mutation event, a parent label gives rise to a derived label
    # Multiple outcomes are possible : 
    # - A derived label survives until present
    # - A derived label becomes a parent label by giving rise to a new derived label and exists in the present-day
    # - A derived label becomes a parent label by giving rise to multiple new derived labels and only none of its descendants
    #   in the present-day carry the label, existing only as an internal label.
    #
    #
    # -- Case 3 : blue label
    # The label "grey" still exists at the present-day because of ancestral retention,
    # and gave rise to "green", "blue" and "purple" label.
    # The label "blue" label deriving from "grey" label is also a parent label for "red" and "yellow" labels
    # "blue" label is not found in its descendants but is part of their evolutionnary history.
    # We have to count for all deriving labels and not only ones on the leaves : they are analoguous in a way to
    # internal lineages of a phylogenetic branch.
    #
    # This example holds a precious insight : a label that is part of the evolutionnary history isn't necesseraly visible
    # at the present-day. When trying to compute oldest and nearest coalescence age for the transition between
    # "grey" --> "blue" labels we need to take all individuals carrying derived labels from "blue", which are "red" and "yellow".
    # To find the nearest/oldest coalescence time between "grey" and "blue" we have to compare pairwise the min and max
    # coalescence times of all "yellow-red" individuals and all "grey" individuals. This is only possible because we are
    # in a infinite allele model.
    #
    # Let's see the transition "grey" -> "blue" :
    # Being part of grey individuals doesn't only mean you still hold a "grey" label. It can also mean you once hold a "grey" label, 
    # that was replaced by another label (or allele, as we can compare the approach to the infinite allele model).
    #
    # Now we have descendant_labels[grey] : {grey, green, yellow, red, blue, purple}
    # And we have descendant_labels[blue] : {blue, red, yellow}
    #
    # To compare the two sides of the transition, we can substract them.
    # derived_side_labels = descendant_labels[blue] = {blue, red, yellow}
    # parent_side_labels = descendant_labels[grey] - descendant_labels[blue] = {grey, purple, green}
    # Therefore : 
    # derived_leaves = [red_ind1, red_ind2, yellow_ind1, yellow_ind2]
    # parent_leaves = [purple_ind1, grey_ind1, grey_ind2, grey_ind3, grey_ind4, green_ind1]
    #
    # Nearest and oldest are then computed as min/max MRCA ages over the complete Cartesian
    # producted of derived_leaves x parent_leaves as mentioned in Bender & Farach-Colton (2000).
    #
    # Couvert et al. (2024) support such a conceptual difference between
    # historical label ancestry and

            
    if debug:
        debug_info = {}

        # ===== BEFORE MERGE =====
        # Species before merge = distinct mutation labels among present-day leaves
        before_species_count = {}
        for leaf in tree.iter_leaves():
            before_species_count[leaf.sp] = before_species_count.get(leaf.sp, 0) + 1

        before_abundances = list(before_species_count.values())

        before_ages = [tree.get_distance(leaf) for leaf in tree.iter_leaves()]
        before_tmrca = max(before_ages) if before_ages else 0.0

        debug_info["before_merge"] = {
            "n_species": len(before_species_count),
            "n_singletons": sum(ab == 1 for ab in before_abundances),
            "abundances": before_abundances,
            "total_abundance": sum(before_abundances),
            "tmrca": before_tmrca,}
        
    
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
                children = node.get_children()
                if len(children) != 2:
                    raise ValueError("The algorithm does not know how to deal with non dichotomic trees.")
                left_species = {leaf.sp for leaf in children[0].iter_leaves()}
                right_species = {leaf.sp for leaf in children[1].iter_leaves()}
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
                    
        
    if spmodel == "paraphyletic" :
        age_column = {
            "mutation": "mutation_coalescence_age",
            "nearest": "nearest_coalescence_age",
            "oldest": "oldest_coalescence_age"
        }[age]
    
        if age in ("nearest", "oldest"):
            node_depth = {id(tree): 0.0} # let's create  a dictionnary with the python id and the depth of each node
            node_age = {id(tree): tree_height} # same with age
            
            for node in tree.traverse("preorder"): # we encounter parents first before children
                if node.up is None: # root has no parents and we already compute 0.0 and tree_height for it
                    continue
                depth = (node_depth[id(node.up)] + float(node.dist)) # recursively from top to bot (preorder) we note depth and age from parent
                node_depth[id(node)] = depth
                node_age[id(node)] = tree_height - depth
            
            # ========================================================================================================
            # Bender & Farach-Colton LCA algorithm
            # ========================================================================================================
            # 
            # h is the topological height of the genealogy : max number of edges between root and leaves.
            # n is the number of nodes.
            # l is the number of leaves.
            # µ is the number of mutations and m is one mutation.
            #
            # Context : 
            # In ETE3 one simple pairwise query LCA(u,v) costs O(h) time where h is the topological
            # height of the genealogy. This means that the bigger the tree is, the more computation time will be needed.
            #
            #   - In a balanced binary tree, a query costs at most O(log₂(l)) because h = log₂(l) levels : each level doubles at each node
            #   - In an unbalanced binary tree, a query costs at most O(l) because h = l levels : each leaf has its own level
            #
            # The problem is that ecophylo has multiple mutations µ that change the leaves identities so we need µ queries.
            # m is one readable mutation on leaves.
            # Let's have the mutation e, splitting a node between a derived D_e side and a parent
            # P_m side so that D_m = 2 leaves and P_m = 2 leaves :
            #
            #            R
            #          /   \* mutation m
            #         A     B
            #        / \   / \
            #       W  X  Y_e Z_e
            #
            # Then if Q_m = 2 x 2 = 4 queries for mutation m ;
            # for every m mutations readable on leaves we have : Q_µ = ∑_m(Q_m) = ∑_m(​∣D_m​∣·​∣P_m​∣) for all µ mutations where D_mi and P_mi
            # aren't equally distributed for each m mutation of µ (some are deep and scan the whole tree, some are recent and scan only two
            # leaves). So Q_µ is a complex number unique for each tree and is not only depending on how many mutations there are.
            # For all Q_µ pairwise comparisons queries it costs :
            #   - O(Q_µ·h)
            #     = O(Q_µ·log₂(l)) in a balanced tree
            #     = O(Q_µ·l) in an unbalanced tree
            #
            # Bender & Farach-Colton show that a LCA (Least Common Ancestor) query on a tree 
            # can be reduced to a RMQ (Range Minimum Query) on the Euler_L array.
            # We can say LCA(u, v) = Euler_E[RMQ_Euler_L(Euler_R[u], Euler_R[v])].
            #
            # Their Lemma 1 states precisely that if RMQ has preprocessing/query
            # complexity <f(n), g(n)>, then LCA on an n-node tree has complexity:
            #
            #             <f(2n - 1) + O(n), g(2n - 1) + O(1)>
            #
            # Because a Euler array is constructed by traversing again and again nodes, it has 2n-1 positions (more later).
            # If the RMQ table is precomputed for every [i,j] with a trivial approach, f(2n - 1) costs O((2n-1)²) = O(n²)
            # because we construct all intervals [i,j] for every starting i and ending j in the Euler_L array.
            # between all positions pairwise. But this allows query time to be O(1).
            # Therefore complexity is : <O(n²) + O(n), O(1)> = <O(n²), O(1)> (because O(n²) >> O(n))
            #
            # Depending on the number of Q_µ, the ETE3 algorithm can be quicker, however, more computing effort will reduce this complexity.
            #
            # Reference:
            # Michael A. Bender and Martin Farach-Colton,
            # "The LCA Problem Revisited",
            # LATIN 2000, Lecture Notes in Computer Science 1776,
            # pp. 88-94, Springer-Verlag, 2000.
                
            euler_E = [] # Euler Tour of size 2n-1 (n is the number of nodes)
            euler_L = [] # Corresponding level of each euler encounter of size 2n-1
            euler_R = {} # Representative array storing each first encounter of each node, size = n
            
            stack = [("visit", tree, 0)] # to store the DFS algorithm advancement needed to complete the Euler Tour
            
            # Let a tree T be : 
            #
            #           R
            #          / \
            #         A   B
            #        / \ / \
            #       W  X Y  Z
            #
            # We want the Euler tour to be E[R A W A X A R B Y B Z B R] and L[0 1 2 1 2 1 0 1 2 1 2 1 0]
            # To achieve that, stack will guide us.
            # Initalizing : stack = [("visit", R, 0)]
            # We remove it and store action, node and level with stack.pop() which removes the last item of the list (compose of one item right now)
            # So we write in Euler_E[R], in Euler_L[0]
            # If we first encounter this node (it's not in R) we add it with its associated position in E
            # Euler_R{R : 0}
            #
            # We check on children of R (which are A and B) to inform the next "stack" loop
            # For each child we add first the parent node but as "return" so we won't add it again in Euler_R
            # And we add the child as "visit" to get its position and add its own children.
            # However, because .pop() removes the LAST item of the stack list, we need to add the NEXT STEP always on the right of the stack list.
            # When processing R, we then want to process A before B and node.get_children() returns [A, B] : that's why we reverse() !
            # Also that's why we place parent-return first before child-visit.
            # 
            # Right now here what stack looks like : 
            # stack = [("return", R, 0), ("visit", B, 1), ("return", R, 0), ("visit", A, 1)] and Euler_E[R], Euler_L[0] and Euler_R{R : 0}
            # Next stack.pop returns ("visit", A, 1) so after doing a loop :
            # stack = [("return", R, 0), ("visit", B, 1), ("return", R, 0), ("return", A, 1), ("visit", X, 2), ("return", A, 1), ("visit", W, 2)]
            # and Euler_E[R A], Euler_L[0 1] and Euler_R{R : 0, A : 1}
            
            # At the end we are left with
            # Euler_E = [R, A, W, A, X, A, R, B, Y, B, Z, B, R]
            # Euler_L = [0, 1, 2, 1, 2, 1, 0, 1, 2, 1, 2, 1, 0]
            # Euler_R = {R : 0, A : 1, W : 2, X : 4, B : 7, Y : 8, Z : 10}
            
            while stack:
                action, node, level = stack.pop()
                if action == "visit":
                    position = len(euler_E)
                    euler_E.append(node)
                    euler_L.append(level)
                    if id(node) not in euler_R:
                        euler_R[id(node)] = position
                    for child in reversed(node.get_children()):
                        stack.append(("return", node, level))
                        stack.append(("visit", child, level+1))
                else:
                    euler_E.append(node)
                    euler_L.append(level)
                    
            # However without further RMQ preprocessing, the minimum has to be found by scanning
            # Euler_L between Euler_R[u] and Euler_R[v]. One scan could cost us O(n) in the worst case where the
            # interval Euler_R[u] and Euleur[v] are the two most far appart leaves of the tree and less on shorter scans.
            # This is not very different from ETE3 algorithm except that we traverse the tree horizontally on an array so we can say that
            # the computation effort for ETE3 is comparable to the Euler scan O(Q_µ·l) ~ O(Q_µ·n).
            #
            # We can build a RMQ table costing O(n²) but making the one scan cost O(1), and for all queries O(Q_µ).
            # The first solution has cheap preprocessing (Euler Tour) and potentially expensive scanning cost (just like ETE3),
            # the RMQ table has expensive preprocessing time but constant query time. Bender & Farach-Colton (2000) now search for
            # an algorithm that keeps O(1) query time but has reduced preprocessing time.
            #
            # The proposed optimisations are
            #       1. a Sparse Table processed in O(n·log₂(n)) time and allowing every query to be
            #          processed in O(1) time so all the algorithm costs <O(n·log₂(n), O(1)> for one scan
            #          and <O(n·log₂(n), O(Q_µ)> for all the pairwise comparison between derived and parent side.
            #       2. a ±1-RMQ array : our L array scanned by RMQ is already a ±1-RMQ array because
            #          all possibles relations movements are toward parent/children (+1 or -1 level).
            #          That special L is called A, the ±1-RMQ array.
            #
            # Instead of storing the position of the minimum of every interval in a matrix RMQ_table[i][j] = argmin(Euler_L[i : j+1]),
            # a Sparse Table stores the position of minima only for intervals of lengths being powers of two so that M[i][k] = argmin(Euler_L[i : i + 2ᵏ].
            # In M, M[i][0] has 2^0 values ; M[i][2] has 2^2 = 4 values and M[i][3] has 2^3 = 8 values.
            # This Sparse Table requires O(n·log₂(n)) preprocessing cost because when building M[i][k] we can use the two already computed
            # intervals of length 2ᵏ⁻¹ forming the left and right halves of the new interval, and we already know there minima so we can
            # deduce the minimum of M[i][k] by comparing minima of M[i][k-1] and M[i + 2ᵏ⁻¹][k-1]. This is dynamic programming.
            # For any query RMQ(i,j) of length L we choose k so that 2ᵏ < L < 2ᵏ⁺¹ two computed blocks.
            # These two intervals may overlap, but together they always cover the query length.
            #
            # We can add optimisation with the unique ±1-RMQ array property to partition the Euler_L array in small blocks of size 
            # log₂(2n - 1)/2. Each block will store a minimum that we will use to build an array of minima (more later).
            # However block_size needs to be chose carfully as too many blocks doesn't reduce the computation time and too large blocks make
            # the computation time of each block too important.
            #
            # This A_prime array will be used to build the Sparse Table.
            
            euler_size = len(euler_L) # Number of positions, for n nodes : 2n-1 positions
            max_block_size = max(1, (math.floor(math.log2(euler_size) / 2))) # max is used to avoid 0 length and floor is used to avoid overtaking euler_size if the last block needs to be smaller.
            n_euler_blocks = math.ceil(euler_size / max_block_size)
            
            A_prime = [] # An array storing minimum level values contained in each block, A_prime[block]
            B = [] # position of the minimum inside blocks, B[block]
            
            for block in range(n_euler_blocks):
                euler_block_start = block * max_block_size # convert block number (because range()) in a global start position in Euler_L.
                euler_block_end = min(euler_block_start + max_block_size, euler_size) # global end position (min() for the final block can't exceed Euler_L)
                euler_block_length = euler_block_end - euler_block_start
                euler_block_min_local_pos = 0 # initiate local position in the block
                for local_pos in range(1, euler_block_length):
                    if (euler_L[euler_block_start + local_pos] < euler_L[euler_block_start + euler_block_min_local_pos]): 
                        euler_block_min_local_pos = local_pos # continue to iterate through the block until we find a minimum value and store its position
                A_prime.append(euler_L[euler_block_start + euler_block_min_local_pos]) # After scanning the block, store its minimum level in A_prime.
                B.append(euler_block_min_local_pos) # After scanning the block, store the position where the minimum occurs.
            
            # Note that we compute one minimum per block. Since the blocks are consecutive and don't overlap, their sum is 2n-1,
            # the Euler_size. This preprocessing scans on 2n-1 positions costs O(2n-1) = O(n) instead of the O(n²) but further preprocessing
            # is needed. Now that we have A_prime ready, we can compute the Sparse Table.
            
            # The Sparse Table M[i][k] stores the position of the minimum in A_prime over the interval starting at i and having length 2**j
            # M[i][j] = argmin(A_prime[i : i + 2ᵏ]. (Remember previously it was argmin(Euler_L[i : i + 2ᵏ], note that A_prime < Euler_L).
            #
            # Why can we use A_prime ? When building M[i][k], any j length spanning across several blocks will store the minimum of that interval of length j.
            # However, this minimum has to be one of the already stored block minima. This means we can use A_prime instead of Euler_L.
            # Note that this is only true for complete blocks, if we start in the middle of a block and end in the middle of a block, we cannot
            # assume yet that A_prime knows these two extrema minimum and cannot correctly compare minima of all covered blocks : it will be handled later.
            #
            # This "block step" also reduces the computation time of the Sparse Table has we now don't compute euler_size = 2n-1 but the
            # number of blocks which is len(A_prime) = euler_size / max_block_size ≈ 2*euler_size / log₂(euler_size) (approx because last block may have a different size)
            # Meaning A_prime contains O(euler_size / log₂(euler_size)) (big-O stills removes constant multiplicator or additions)
            # Then if the Sparse Table takes O(x log₂(x)) where x is A_prime values then it takes
            #       O((euler_size/log(euler_size)) * log(euler_size/log(euler_size)))
            #     = O(euler_size) = O(n)
            # For now we build A_prime for O(n) and M[i][k] for O(n) ; O(2n) = O(n) (big-O stills removes constant multiplicator or additions)
            #
            # Lets take A_prime = [7, 4, 6, 2, 8, 3, 5, 1], len(A_prime) is 8
            # As explained before, M[i][k] stores the position of the minimum in A_prime over an interval starting at i of length 2ᵏ.
            # We have the query [2, 7] :
            # positions:          0  1  2  3  4  5  6  7
            # A_prime:           [7, 4, 6, 2, 8, 3, 5, 1]
            #                           |---------------|
            #                              query [2, 7]
            # The query contains 7-2+1 = 6 values
            # We choose k = floor(log₂(6)) = 2 ; Therefore 2ᵏ = 4
            # We use two already precomputed power-of-two intervals of length 4 :
            #       first interval  = positions [2, 5] = [6, 2, 8, 3]
            #       second interval = positions [4, 7] = [8, 3, 5, 1]
            # For the Sparse Table to returns these :
            #       first = M[2][2]
            #       second = M[4][2]
            # followed by : min(A_prime[first], A_prime[second]) returning the position corresponding to the smaller value.
            # So one RMQ does not scan its 6 values anymore and apply a unique min() : it uses two table lookups and one comparison,
            # on two already computed min(), giving O(1) query time instead of O(n).
            
            M_levels = math.floor(math.log2(len(A_prime))) + 1 # Number of powers 2ᵏ required to represent every possible A_prime interval length
            M = [[None] * M_levels for _ in range(len(A_prime))] # Allocate one Sparse-Table row per A_prime position, and one column per 2ᵏ interval size
            for i in range(len(A_prime)): # Initialise the first column
                M[i][0] = i # k=0 is an interval of 1 which the minimum is necessarily at position i
            for k in range (1, M_levels): # Now we construct the following columns recursively from the previous one (dynamic code)
                interval_length = 2**k
                half_length = 2**(k - 1) 
                # All interval_length starting at i must remain completely inside A_prime
                # If len(A_prime) then the last valid starting point for EVERY interval_length is len(A_prime) - interval_length
                # so 'i' should range from 0 to (len(A_prime) - interval_length) + 1 (range exludes upper bound)
                # Now in this for loop we're doing dynamic programming as explained above
                # For the column k, the previous column k-1 already stores the minimum position of the left half of k which begins at i and has half_length (2**j-1)
                # The right half begins at half_length position after i and M already knows its minimum because it has length 2ᵏ⁻¹.
                for i in range(len(A_prime) - interval_length + 1):
                    left_min = M[i][k - 1]
                    right_min = M[i + half_length][k - 1]
                    if A_prime[left_min] <= A_prime[right_min]: # Now we compare tha A_prime values of the two previous columns to store the position of the minimum of this new column, that is a glued version of the two previous columns of half_length.
                        M[i][k] = left_min
                    else:
                        M[i][k] = right_min
            
            # The Sparse-Table M can now answer RMQ queries over complete blocks through A_prime. But an arbitrary query does not
            # necessarily begin at the beginning of a block or finish at the end of a block. This is the extrema problems explained above.
            # For example : the query RMQ(i,j)
            #       [ 0 1 2 3 2 ] [ 3 4 3 2 1 ] [ 2 3 4 3 2 ] = Euler_L separated in blocks
            #            |---------------------------|
            #            i                           j
            # We can query the middle block with A_prime and M but not the beginning and ending of the query.
            # Observation 3 states that adding or subtracting the same constant from every element of an array does not change its RMQ positions.
            #
            # For example we can normalise a block by substracting the first value by itself, making it 0 : 
            #
            #       original_block 0   = [7, 8, 9, 8, 7] ; min is the first and 5th value
            #       normalised_block 0 = [0, 1, 2, 1, 0] ; min is the first and 5th value
            #
            #       original_block 1   = [3, 4, 3, 2, 1] ; min is the 5th value
            #       normalised_block 1 = [0, 1, 0,-1,-2] ; min is the 5th value
            #
            #       original_block 2   = [2, 3, 4, 3, 2] ; min is the first and 5th value
            #       normalised_block 2 = [0, 1, 2, 1, 0] ; min is the first and 5th value
            #
            # As you can see, the minimum of original and normalised are in the same position despite the transformation.
            # Because Euler_L is a ±1-array we can describe a normalised block by its internal movements :
            #
            #       normalised_block 0 = [0, 1, 2, 1, 0]
            #                              +1 +1 -1 -1
            #       movement_array 0   = [ 1, 1, 0, 0]
            #
            #       normalised_block 1 = [0, 1, 0,-1,-2]
            #                              +1 -1 -1 -1
            #       movement_array 1   = [ 1, 0, 0, 0]
            #
            #       normalised_block 2 = [0, 1, 2, 1, 0]
            #                              +1 +1 -1 -1
            #       movement_array 2   = [ 1, 1, 0, 0]
            #
            # Here +1 is encoded by 1 and -1 by 0.
            #
            # Why is it useful to normalise blocks ? Because it can reduce their nnumber. See original_block 0 and original_block 2, they are
            # both different but still have the same normalised form. Therefore we can use the same normalised block type to get the same
            # position information, and to retrieve different Euler value (see that 7 != 2).
            # How many different block types can exist ?
            # We have two different possibilities : 1 and 0. In a block there are b values (max_block_size) and there are b-1 movements.
            # Therefore we can except 2ᵇ⁻¹ different possible movement_arrays (normalised block types). Because we defined b as
            # log₂(euler_size)/2 = log₂(2n - 1)/2 then this computation effort of normalised blocks is
            #       2**(b - 1)
            #       = 2**((log₂(2*n - 1) / 2) - 1)
            #       = sqrt(2*n - 1) / 2
            #       = O(sqrt(n))

            
            
            
            micro_tables = {} # store one complete in-block RMQ table for every possible normalised ±1 block type.
        
            for type_length in range(1, max_block_size + 1): # Consider every possible block length; the final Euler_L block may be shorter then max_block_size
                movement_arrays = [()]
                for step in range(type_length - 1):
                    next_possible_movement_arrays = []
                    for movements in movement_arrays:
                        next_possible_movement_arrays.append(movements + (1,))
                        next_possible_movement_arrays.append(movements + (-1,))
                    movement_arrays = next_possible_movement_arrays
                    
                    # Recursively,
                    # movement_arrays[(1),
                    #                 (-1)]
                    #
                    # movement_arrays[(1,1),
                    #                 (1,-1),
                    #                 (-1,1),
                    #                 (-1,-1)]
                    #
                    # For each step, doubling the possibilities with either +1 or -1 at the end (binary)
                    
                for movements in movement_arrays: # movements being each (-1,1,1,-1) blocks of movement_arrays 
                    normalised_type = [0] # Normalised blocks always start at zero
                    for movement in movements: # movement being each 1 or -1 inside a block of movement_arrays
                        normalised_type.append(normalised_type[-1] + movement) # append the previous_value + the movement (+1 or -1) so if previous value is 2 thenwe append either 3 or 1 depending on the movement array we are precomputing
                    
                    micro_table = [[None]*type_length for _ in range(type_length)] # precompute every possible RMQ interval inside this block type
                                                                               # table[left][right] stores the local position of the minimum
                                                                               # of each query for the normalised block to which the table is built from.
                                                                               # micro_table =
                                                                               #         right
                                                                               #         0     1     2     3     4
                                                                               #  left 0 None  None  None  None  None
                                                                               #       1 None  None  None  None  None
                                                                               #       2 None  None  None  None  None
                                                                               #       3 None  None  None  None  None
                                                                               #       4 None  None  None  None  None
                    for left in range(type_length): # for each line of table
                        min_pos = left # left is the minimum
                        micro_table[left][left] = left # [left][left] is a 1 length query : it has to be its own min
                        for right in range(left+1, type_length): # now we iter through columns of the line "left"
                            if normalised_type[right] < normalised_type[min_pos]: # if we find a smaller value within the normalised block for this line
                                min_pos = right  # then we store this new minimum
                            micro_table[left][right] = min_pos
                    micro_tables[(type_length, movements)] = micro_table
                    
                    # let's take this array :
                    # positions:      0  1  2  3  4
                    # values:        [0, 1, 2, 1, 0]
                    #                          ^
                    #                        minimum
                    #
                    # left = 0 then for interval table[0][0] the minimum is located on... 0.
                    # then we iter through all columns of line left=0, columns are called "right"
                    # right = 1 -> interval [0,1] = [0,1] --> min still in position 0
                    # right = 2 -> interval [0,2] = [0,1,2] --> min still in position 0
                    # right = 3 -> interval [0,3] = [0,1,2,1] --> min still in position 0
                    # right = 4 -> interval [0,4] = [0,1,2,1,0] --> min still in position 0
                    # micro_table =
                    #         right
                    #         0     1     2     3     4
                    #  left 0 0     0     0     0     0
                    #       1 None  None  None  None  None
                    #       2 None  None  None  None  None
                    #       3 None  None  None  None  None
                    #       4 None  None  None  None  None
                    # left = 1
                    # [1,1] = [1]       -> min still in position 1
                    # [1,2] = [1,2]     -> min still in position 1
                    # [1,3] = [1,2,1]   -> min still in position 1
                    # [1,4] = [1,2,1,0] -> min is now in position 4
                    #         right
                    #         0     1     2     3     4
                    #  left 0 0     0     0     0     0
                    #       1 None  1     1     1     4
                    #       2 None  None  None  None  None
                    #       3 None  None  None  None  None
                    #       4 None  None  None  None  None
                    
                    
                # We have already built one micro_table for every possible normalised block type.
                # We now need to inspect Euler blocks of Euler_L to determine its precomputed block type.
                # euler_block_types[euler_block_idx] will store the block type of the "block" euler_block as (euler_block_length, movements)
                # Therefore when a query starts or ends in the middle of this Euler_L block = [7, 8, 9, 8, 7] (the 7th lets say)
                # we can euler_block_types[7] and retrieve = (5, (1, 1, -1, -1)) this is a key to find local minima in micro_tables :
                # micro_tables[(5, (1, 1, -1, -1))]

            euler_block_types = [] 
            for euler_block_idx in range(n_euler_blocks):
                euler_block_start = euler_block_idx * max_block_size # convert into global start position in the Euler_L array
                euler_block_end = min(euler_block_start + max_block_size, euler_size) # convert into global end position in the Euler_L array ; min avoids extending further than the limit of euler_size if last euler block is smaller
                euler_block_length = euler_block_end - euler_block_start
                normalised_euler_block = [euler_L[euler_pos] - euler_L[euler_block_start] for euler_pos in range(euler_block_start, euler_block_end)]
                # for each Euler_block, build a normalised version by removing euler_L[euler_block_start] value (the value of the first position) for
                # every euler_pos inside that block ; this is literally the normalisation described
                # above as euler_L[euler_pos] - euler_L[euler_block_start] = 0 when euler_pos = 0.
                #    Euler_L block:
                #       [7, 8, 9, 8, 7] ; euler_L[euler_block_start] = 7
                #    subtract 7 from every value:
                #       [7-7, 8-7, 9-7, 8-7, 7-7]
                #    normalised_euler_block:
                #       [0, 1, 2, 1, 0]
                
                euler_block_movements = []
                # Now we describe the movement for each euler_block
                for local_pos in range(1, euler_block_length):
                    movement = (normalised_euler_block[local_pos] - normalised_euler_block[local_pos - 1])
                    # normalised_euler_block = [0, 1, 2, 1, 0]
                    # local_pos = 1:
                    #       1 - 0 = +1
                    # local_pos = 2:
                    #       2 - 1 = +1
                    # local_pos = 3:
                    #       1 - 2 = -1
                    # local_pos = 4:
                    #       0 - 1 = -1
                    if movement == 1:
                        euler_block_movements.append(1)
                    elif movement == -1:
                        euler_block_movements.append(-1)
                    else:
                        raise RuntimeError("Euler_L doesn't satisfy the ±1 property")
                euler_block_movements = tuple(euler_block_movements) # because we will use it as a key of the micro_tables dictionnary : lists cannot be used as keys
                euler_block_types.append((euler_block_length, euler_block_movements))

                # Example :
                #
                #       local positions:       0  1  2  3  4
                #       Euler_L values:       [3, 4, 3, 2, 1]
                #                                          ^
                #                                         min
                #       normalised : [0, 1, 0, -1, -2]
                #       movements :  [1, -1, -1, -1]
                #       length = 5
                #       micro_tables_key = (5,(1,-1,-1,-1))
                #
                # Suppose an RMQ query reaches this last block but ENDS at its third value:
                #
                #       local positions:       0  1  2  3  4
                #       Euler_L values:       [3, 4, 3, 2, 1]
                #                              |-----|
                #                              queried part
                # We already precomputed the min of this block as being in the 5th position, but the query ending in the middle doesn't align
                # with the precomputed answer for that block size, we need local minima informations stored in micro_tables :
                #       micro_tables[(5,(1,-1,-1,-1))][0][2]
                #                             │        │  │
                #                             │        │  └──────── end of the queried interval inside the block (column // right)
                #                             │        └─────────── start of the queried interval inside the block (row // left)
                #                             └──────────────────── choose the correct precomputed block type micro_table (left/right table)


        # ========================================================================================================
        # Preprocessing summary ; with an example of a tree with 50 000 leaves
        # ========================================================================================================
        #
        # l = 50000 ; n = 99999 nodes ; euler_size = 2n-1 = 199997 positions
        #
        # 1. Compute node_depth and node_age
        #       Time
        #       Storage
        # 2. Construct Euler_E, Euler_L and Euler_R
        # 3. Divide Euler_L into small Euler blocks and construct a A_prime minima values array and B minima position array 
        # 4. Build Sparse Table M
        # 5. Precompute ±1 micro_tables
        
        
            # ========================================================================================================
            # Retrieve all comparisons that will be queried
            # ========================================================================================================
            
            all_labels = {1} # set (no double values)
            for row in mutation_table:
                all_labels.add(row["label"])
                all_labels.add(row["parent_label"]) # ensure we have ALL labels, without doubles as we store them in a set
                # all_labels = {grey, blue, purple, red, yellow, green}
            descendant_labels = {label: {label} for label in all_labels} # add own label as descendant, grey : {grey} means grey descends from grey. This is initialisation.
            for row in reversed(mutation_table):
                descendant_labels[row["parent_label"]].update(descendant_labels[row["label"]])
            # mutation_table was created in historical order, we need to reverse it so that all descendants of a label are known when updating its lineage.
            # here's the mutation table of the general tree example:
            #
            #       "parent_label" -> "label"
            #       grey -> blue
            #       grey -> purple
            #       blue -> red
            #       blue -> yellow
            #       grey -> green
            #
            # in reverse we inspect grey -> green first, so for descendant_label[grey] we add the descendant_labels[green] lineage.
            # Why lineage and not only "green" because if green had descendants added previously to its lineage, then it means that they also
            # descend from grey.
            # descendant_labels = {
            #     "grey":   {"grey", "green"},
            #     "blue":   {"blue"},
            #     "purple": {"purple"},
            #     "red":    {"red"},
            #     "yellow": {"yellow"},
            #     "green":  {"green"}}
            # Then we add yellow and red to blue, just like green these two labels have no descendant labels so its just like if we add only them
            # descendant_labels = {
            #     "grey":   {"grey", "green"},
            #     "blue":   {"blue", "yellow", "red"},
            #     "purple": {"purple"},
            #     "red":    {"red"},
            #     "yellow": {"yellow"},
            #     "green":  {"green"}}
            # Then grey -> purple and grey -> blue :
            #     "grey":   {"grey", "green", "purple", "blue", "yellow", "red"},
            #     "blue":   {"blue", "yellow", "red"},
            #     "purple": {"purple"},
            #     "red":    {"red"},
            #     "yellow": {"yellow"},
            #     "green":  {"green"}}
            # As you can see we added ALL blue descendants as also descendants of grey.
            # So when querying descendant_labels[blue] = "blue", "red", "yellow"
            # descendant_labels[grey] = "grey", "green", "purple", "blue", "yellow", "red"
    
            label_to_leaves = {}
            for leaf in tree.iter_leaves():
                label_to_leaves.setdefault(leaf.sp, []).append(leaf)
                # label_to_leaves = {
                #   grey:   [grey_ind1, grey_ind2...],
                #   red:    [red_ind1...],
                #   yellow: [yellow_ind1...]}
                # blue has no leaves in present, so it's not part of this dictionary
            for row in mutation_table:
                derived_label = row["label"]
                parent_label = row["parent_label"]
                derived_side_labels = descendant_labels[derived_label]
                parent_side_labels = (descendant_labels[parent_label] - derived_side_labels)
                # when comparing grey -> blue event ; compare derived_side of blue being descendant_labels[blue] = {"blue", "red", "yellow"}
                # and parent_side of grey being descendant_labels[grey] = {"grey", "green", "purple", "blue", "yellow", "red"} without
                # its blue lineage (so - {"blue", "red", "yellow"}) which is {"grey", "green", "purple"}
                
                # However, to compute a LCA pairwise of individuals descending from blue, we need to retrieve de leaves : but blue has no living leaves !
                
                derived_leaves = []
                for label in derived_side_labels:
                    derived_leaves.extend(label_to_leaves.get(label, []))
                    # Translate labels in leaves or in empty list :
                    # derived_side_labels = {"blue", "red", "yellow"}
                    # label = blue ; derived_leaves = []
                    # label = red ; derived_leaves = [red_ind1, red_ind2]
                    # label = yellow ; derivied_leaves = [yellow_ind1, yellow_ind2]
                    # Therefore, derived_leaves are [red_ind1, red_ind2, yellow_ind1, yellow_ind2]
                    # no problem with blue.
                
                parent_leaves = []
                for label in parent_side_labels:
                    parent_leaves.extend(label_to_leaves.get(label, []))
                    
                if not derived_leaves or not parent_leaves:
                    raise RuntimeError("Unexpected empty descendant set for mutation "
                                       f"{parent_label} -> {derived_label} :"
                                       f"derived_leaves = {len(derived_leaves)},"
                                       f"parent_leaves = {len(parent_leaves)}")
                
                # if either one side is empty (shouldn't happen)
                
                nearest_age = None
                oldest_age = None
                
                for derived_leaf in derived_leaves:
                    derived_euler_pos = euler_R[id(derived_leaf)] # get the first euler tour occurrence of the derived leaf
                    for parent_leaf in parent_leaves:
                        parent_euler_pos = euler_R[id(parent_leaf)] # get the first euler tour occurrence of the parent leaf
                        if derived_euler_pos <= parent_euler_pos: # if derived position is smaller or = to parent position
                            left_query_euler_pos = derived_euler_pos # left pos will be derived position
                            right_query_euler_pos = parent_euler_pos # right pos will be parent position
                        else: # if dervied position is greater than parent position
                            left_query_euler_pos = parent_euler_pos # left pos will be parent position
                            right_query_euler_pos = derived_euler_pos # right pos will be derived position
                            # Because RMQ(left_pos, right_pos) where left_pos <= right_pos
                            
                        left_euler_block_idx = (left_query_euler_pos // max_block_size)
                        right_euler_block_idx = (right_query_euler_pos // max_block_size)
            
                        # CASE 1 : both RMQ endpoints are inside the SAME Euler block
                        if left_euler_block_idx == right_euler_block_idx: # then the interval is already precomputed in the microtable of this block
                            euler_block_start = (left_euler_block_idx * max_block_size) # get global block start
                            local_left = (left_query_euler_pos - euler_block_start) # convert global startpoint of the query in local startpoint
                            local_right = (right_query_euler_pos - euler_block_start) # convert global endpoint of the query in local endpoint
                            micro_tables_key = (euler_block_types[left_euler_block_idx]) # get the microtable key from the block (right == left)
                            local_min_pos = micro_tables[micro_tables_key][local_left][local_right] # get local min pos from microtables with the key corresponding to our block, and the cover local_left-local_right
                            minimum_euler_pos = (euler_block_start + local_min_pos) # get global min pos
                        
                        # CASE 2 : RMQ endpoints are located in different euler blocks
                        # even if the queries cover the complete block on the endpoints, we still scan them
                        # locally because it's the same result, but holistic approach
                        else: 
                            # 2A : minimum inside the left partial Euler block
                            left_euler_block_start = (left_euler_block_idx * max_block_size) # get global block start
                            left_euler_block_end = min(left_euler_block_start + max_block_size, euler_size) # get global block end
                            left_euler_block_length = (left_euler_block_end - left_euler_block_start) # get block length
                            local_left = (left_query_euler_pos - left_euler_block_start) # get local leaf position by substracting global start and global left_query_euler_pos
                            micro_tables_key = (euler_block_types[left_euler_block_idx]) # get the micro_table of that block
                            left_local_min_pos = micro_tables[micro_tables_key][local_left][left_euler_block_length - 1] # for the micro_table assigned to the key, get the min pos between local_left and the remaining of the block length
                            left_min_euler_pos = (left_euler_block_start + left_local_min_pos) # get global min pos
                            # 2B : minimum inside the right partial Euler block
                            right_euler_block_start = (right_euler_block_idx * max_block_size) # same but for right block
                            local_right = (right_query_euler_pos - right_euler_block_start)
                            micro_tables_key = (euler_block_types[right_euler_block_idx])
                            right_local_min_pos = micro_tables[micro_tables_key][0][local_right] # for the right extremity, it always locally starts from 0 and stops at local_right
                            right_min_euler_pos = (right_euler_block_start + right_local_min_pos)
                            
                            if (euler_L[left_min_euler_pos] <= euler_L[right_min_euler_pos]): # check which is min
                                minimum_euler_pos = left_min_euler_pos
                            else:
                                minimum_euler_pos = right_min_euler_pos
                            
                            # 2C : minimum inside the complete Euler blocks located between two partial boundary blocks
                            #
                            #
                            middle_left_euler_block_idx = (left_euler_block_idx + 1) # the middle part starts immediatly after the left partial block ; we have the id of the leftest middle block
                            middle_right_euler_block_idx = (right_euler_block_idx - 1) # the middle part ends just before the right partial block ; we have the id of the rightest middle block
                            if middle_left_euler_block_idx <= middle_right_euler_block_idx: # Is there any block between these two extreme middle blocks ?
                                middle_block_count = (middle_right_euler_block_idx - middle_left_euler_block_idx + 1) # How many complete blocks ?
                                k = math.floor(math.log2(middle_block_count)) # let's choose the power of two that is not greater than middle_block_count
                                first_middle_candidate_euler_block_idx = M[middle_left_euler_block_idx][k] # Get the first A_prime min candidate starting from middle_left_euler_block_idx first' position and containing 2^k blocks
                                second_middle_interval_start = (middle_right_euler_block_idx - 2**k + 1) # Get the second interval starting point from "middle_right_euler_block_idx - 2**k + 1" which is its endpoint.
                                
                                # left  B4   B5  B6   B7   B8   B9  right    Block_idx
                                #     [ 8 ][ 5 ][ 7 ][ 3 ][ 6 ][ 4 ]         A_prime min of each block
                                #  |-----------------------------------|     query
                                #
                                # This query covers 6 blocks : so we assign it 2**k ≤ 6 < 2**k+1 <=> 4 ≤ 6 < 8 so that k = 2
                                # The fist middle interval is therefore 4 blocks long and starting from B4 to B7
                                # The second middle interval must also contain 2**k blocks (4 blocks here) and must end at middle_right_euler_block_idx (end = 9 = B9)
                                # start = end - length + 1 <=> start = 9 - 4 + 1 = 6
                                # left  B4   B5    B6   B7   B8   B9 right
                                #      [ 8 ][ 5 ][ 7 ][ 3 ][ 6 ][ 4 ]
                                #       |------------------|             first_middle_interval
                                #                   |------------------| second_middle_interval
                                #                   ^^^^^^^^^
                                #                    overlap ; not problematic
                                
                                second_middle_candidate_euler_block_idx = M[second_middle_interval_start][k] # Get the second A_prime min candidate
                                if (A_prime[first_middle_candidate_euler_block_idx] <= A_prime[second_middle_candidate_euler_block_idx]):
                                    middle_min_euler_block_idx = (first_middle_candidate_euler_block_idx)
                                else:
                                    middle_min_euler_block_idx = (second_middle_candidate_euler_block_idx)
                                if (A_prime[middle_min_euler_block_idx] < euler_L[minimum_euler_pos]): # if the middle min stored in A_prime is smaller than the left and right partial min
                                    minimum_euler_pos = (middle_min_euler_block_idx * max_block_size + B[middle_min_euler_block_idx])
                                    # replace the previous min and convert it in global euler_L pos, we need to do the general multiplication
                                    # block_id * max_block_size but we also need to add the local position of the minima in the block stored in B[]
                        
                        mrca_node = euler_E[minimum_euler_pos]
                        mrca_age = node_age[id(mrca_node)]
                        
                        if nearest_age is None or mrca_age < nearest_age:
                            nearest_age = mrca_age
                        if oldest_age is None or mrca_age > oldest_age:
                            oldest_age = mrca_age
    
                row["nearest_coalescence_age"] = nearest_age
                row["oldest_coalescence_age"] = oldest_age
        
        
        # ========================================================================================================
        # Reconstruct the paraphyletic label-tree
        # ========================================================================================================     

        lineage = {}  # Map each currently reconstructed historical label to its phylogenetic subtree. Dictionary of subtrees associated to each label.
        
        # Classic tree reconstruction juste like other spmodel trees
        for leaf in tree.iter_leaves():
            label = leaf.sp
            if label not in lineage:
                popInd = [0]*(ndeme+1)
                popInd[leaf.deme] = 1
                
                new_leaf = type(tree)() # creates a new empty ETE3 node
                new_leaf.add_features(sp = label,
                                      name = "sp" + str(label),
                                      popInd = popInd,
                                      mergedInd = str(leaf.name),
                                      age = 0.0)
                lineage[label] = new_leaf
            else:
                lineage[label].popInd[leaf.deme] += 1 # if the traversed leaf's label is already in lineage, it means we can just add 1
                                                      # to the popInd value of lineage[label].popInd and in the precise deme given by the traversed leaf
                lineage[label].mergedInd += " " + str(leaf.name)


        innerNodeIndex = 0 # node index
        present_labels = set(lineage.keys())
        
        ordered_mutations = [row for _, row in sorted(enumerate(mutation_table),
                                                      key = lambda item:(
                                                          item[1][age_column],
                                                          -item[0]))]                 # mutation_table may be ordered historically,
                                                                                      # this order may have changed in nearest or oldest cols
        for event in ordered_mutations:
            parent_label = event["parent_label"]
            derived_label = event["label"]
            event_age = float(event[age_column])
            
            if derived_label not in lineage:
                raise RuntimeError("A derived label isn't in lineage dictionary, either a leaf is missing from lineage"
                                   f" or a transition label {parent_label} -> {derived_label} is computed as derived_label"
                                   " before being computed as parent_label : this shouldn't be, computing it as parent first"
                                   " as intended should assign it a descendant subtree in the dictionary, report this error"
                                   " and the code with a seed.")
            
            derived_tree = lineage[derived_label] # retrieve the descendants history (subtree) of the derived label
            
            if parent_label not in lineage: # CASE 1 : parent_label doesn't exist yet in lineage dictionary : it is a transition label
                lineage[parent_label] = derived_tree # assign it the descendants history ; for blue -> red, blue isn't in lineage yet so
                continue                             # assign lineage[blue] = lineage[red] ; when computing blue -> yellow we will now be in
                                                     # CASE 2.
                                                     
            parent_tree = lineage[parent_label] # CASE 2 : if parent is in lineage, retrieve its subtree
            
            if (not parent_tree.is_leaf() and abs(parent_tree.age - event_age) < 1e-12): # CASE 2A : an internal node already exist at exactly "event_age" --> reuse it and create/extend the polytomy
                # Let's take our pilot example with these events in "oldest" :
                # blue -> red = 50
                # blue -> yellow = 50
                # grey -> blue = 100
                # grey -> purple = 100
                # grey -> green = 100
                # right now parent_tree looks like :
                # grey -> blue is processed like this, see more in CASE 2B
                #                         ┌──────────── grey
                #                    ─────┤ age 100
                #                         │      ┌───── red
                #                         └──────┤ age 50
                #                                └───── yellow
                # then we process grey -> purple and notice it has the same node, therefore we need to create a polytomy :
                #                         ┌──────────── grey
                #                         │
                #                    ─────┼──────────── purple
                #                         │
                #                         │      ┌───── red
                #                         └──────┤
                #                                └───── yellow
                
                replacement_node = parent_tree # replacement_node refers to the already existing subtree, we will complete it later after else CASE 2B
            else: # CASE 2B : parent_label already has a subtree in lineage and if parent_tree is still a leaf in dictionary (first occurrence of grey as parent_label for example)
                  # Because we checked parent_tree.age - event_age : a leaf has .age at 0.0
                replacement_node = type(tree)()
                replacement_node.name = "n" + str(innerNodeIndex)
                innerNodeIndex += 1
                
                replacement_node.add_features(sp = None,
                                              age = event_age) # new abs age
                
                parent_tree.dist = event_age - parent_tree.age # new dist between the parent and child
                if parent_tree.dist < -1e-12:
                    raise RuntimeError("Historical reconstruction produced a negative branch "
                                       f"for parent side of {parent_label} -> {derived_label}: "
                                       f"event_age = {event_age}, "
                                       f"parent_age = {parent_tree.age}")
                if parent_tree.dist < 0: 
                    parent_tree.dist = 0.0
                
                replacement_node.add_child(parent_tree)
            
            # Attach derived subtree to the node
            
            derived_tree.dist = event_age - derived_tree.age
            if derived_tree.dist < -1e-12:
                raise RuntimeError("Historical reconstruction produced a negative branch "
                                   f"for derived side of {parent_label} -> {derived_label}: "
                                   f"event_age = {event_age}, "
                                   f"derived_tree.age = {derived_tree.age}")
            
            if derived_tree.dist < 0:
                derived_tree.dist = 0.0
                
            if derived_tree not in replacement_node.get_children():
                replacement_node.add_child(derived_tree)
                
            lineage[parent_label] = replacement_node
            
        if 1 not in lineage:
            raise RuntimeError("Historical reconstruction did not recover ancestral label 1")
        crown_tree = lineage[1]
        root_stem_length = tree_height - float(crown_tree.age)
        if root_stem_length < -1e-12:
            raise RuntimeError("Reconstructed tree is older than the original genealogy : error")
        if root_stem_length < 0:
            root_stem_length = 0.0 # float miscalculation error
        if root_stem_length > 1e-12:
            ancestral_root = type(crown_tree)()
            crown_tree.dist = root_stem_length
            ancestral_root.add_child(crown_tree)
            ancestral_root.dist = 0.0
            tree = ancestral_root
        else:
            crown_tree.dist = 0.0
            tree = crown_tree
        
        reconstructed_labels = [leaf.sp for leaf in tree.iter_leaves()]
        
        if (len(reconstructed_labels) != len(present_labels) or set(reconstructed_labels) != present_labels):
            raise RuntimeError(
            "Historical reconstruction changed the present-day label set: "
            f"expected {sorted(present_labels)}, "
            f"obtained {sorted(reconstructed_labels)}."
        )

    if spmodel in ("loose", "lacy"):
        nsp = 1
        for leaf in tree.iter_leaves():
            leaf.name = "sp"+str(nsp)
            nsp += 1
    if spmodel == "genealogy":
        print("Warning : This output is a genealogical tree without phylogenetic attributes. It may constain paraphyletic relations arising from ancestral retention")
        print("Please don't try to getLTT() on this output as this you will count multiple branches from the same lineage.")
            
            
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
    
    tree.add_features(
        spmodel=spmodel,
        mu=mu,
        tau=tau,
        mutation_table = mutation_table,
        age_convention = age)
    
    if debug:
        # ===== AFTER MERGE =====

        if spmodel == "genealogy":
            after_species_count = {}
            for leaf in tree.iter_leaves():
                after_species_count[leaf.sp] = after_species_count.get(leaf.sp, 0) + 1

            after_abundances = list(after_species_count.values())

        else:
            # In loose/lacy, after merge, each remaining leaf is one species.
            after_abundances = []

            for leaf in tree.iter_leaves():
                if hasattr(leaf, "popInd"):
                    after_abundances.append(sum(leaf.popInd))
                elif hasattr(leaf, "mergedInd"):
                    after_abundances.append(len(leaf.mergedInd.split()))
                else:
                    after_abundances.append(1)

        after_ages = [tree.get_distance(leaf) for leaf in tree.iter_leaves()]
        after_tmrca = max(after_ages) if after_ages else 0.0

        debug_info["after_merge"] = {
            "n_species": len(after_abundances),
            "n_singletons": sum(ab == 1 for ab in after_abundances),
            "abundances": after_abundances,
            "total_abundance": sum(after_abundances),
            "tmrca": after_tmrca,}

        return tree, debug_info

    return tree

def ubranch_mutation(node, mu, tau = 0, seed = None):
    """
    Draw the first mutation event on a branch of length B with a latent period
    of time tau defining a mutable length as :
        mutable_length = max((B - tau), 0)
    
    The waiting time of the first event is drawn from an exponential
    distribution with rate mu : 
        Tmut ~ Exponential(rate = mu)
        with density f(t) = mu * exp(-mu * t) with t ≥ 0
    
    Logic
    ----------
    If Tmut > mutable_length then we did not draw a mutation event on
    that branch. If Tmut ​≤ mutable_length then the first event occurs on the
    branch. Then position on the whole branch of length B is
        mut_from_parent = tau + Tmut
    
    
    Mathematical proof of consistency with Poisson Process
    ----------
    A Poisson process with rate mu means that mutations occur randomly along a
    branch with average rate mu per generation.
    
    On the mutable_length of B, the number of mutation follows :
        N ~ Poisson(mu * mutable_length)
    Therefore :
        P(N = k) = ((mu * mutable_length)^k / k!) * exp(-mu * mutable_length)
    And the probability of zero mutations (k = 0) is :
        P(N = 0) = exp(-mu * mutable_length)
    Therefore the probability to see at least one mutation is :
        P(N ​≥ 1) = 1 - P(N = 0)
                   1 - exp(-mu * mutable_length)
    
    P(Tmut ​≤ mutable_length) is the area under the density between 0 and
    mutable_length so that : 
        P(Tmut ​≤ mutable_length) = ∫[0,mutable_length] mu * exp(-mu * t) dt
        P(Tmut ​≤ mutable_length) = 1 - exp(−mu * mutable_length)
    
    Then P(N ​≥ 1) = P(Tmut ​≤ mutable_length)
    
    So drawing Tmut and checking whether Tmut <= mutable_length is equivalent
    to checking whether the Poisson process produced at least one mutation on
    the mutable part of the branch. With the advantage that the waiting time
    gives us a time of where the first mutation is drawn.

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
    tuple
        (has_mutation, mut_from_parent)
        
        has_mutation : bool
            True if a mutation occured (Poisson(lambd) >= 1)
        
        mut_from_parent : float or None
            Distance from the parent node to the mutation. None if no mutation
            occured.

    Examples
    --------
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
    
    mutable_length = max(node.dist - tau, 0) # compute the length where mutation can happen (imagine a latent length where no mutation can happen between parent and mutation of length tau)
      
    if mutable_length == 0 or mu == 0: # no mutations can arise
        return False, None
    
    Tmut = np.random.exponential(scale = 1 / mu) # the waiting time of the first event of a Poisson Distribution as 1/scale * exp(-x/scale)
                                                 # we defined scale as 1/mu so Tmut is sampled in 1/(1/mu) * exp(-x/(1/mu))
                                                 # But 1/(1/mu) = mu and -x/(1/mu) = -x*mu    so Tmut is sampled in mu*exp(-x*mu)
    
    if Tmut > mutable_length: # the waiting time is further the mutable_length
        return False, None
        
    mut_from_parent = tau + Tmut # the case where Tmut is < mutable_length is when Poisson(Lambda) >= 1 ; we compute tau + Tmut to make the label_tree

    return True, mut_from_parent


if __name__ == "__main__":
        import doctest
        doctest.testmod()