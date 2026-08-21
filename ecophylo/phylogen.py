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

                              ┌───── grey_ind1
                         ┌────┤n4
                         │    └───── grey_ind2
                  ┌──────┤n2
                  │      │        ┌─ red_ind1
                  │      └*red────┤n5
                  │               └─ red_ind2
           ─*grey─┤n1
                  │         ┌─────── grey_ind3
                  │      ┌──┤n6
                  │      │  └*yellow─ yellow_ind1
                  └──────┤n3
                         │  ┌─────── grey_ind4
                         └──┤n7
                            └─────── grey_ind5
        
        100     75    50    25    0 BP
    
        We can see here that all individuals of two species don't share the
        same common ancestor : in the relation red and grey, red_ind1 and
        grey_ind1 have a common ancestor at node n2. But red_ind1 and grey_ind5
        have a common ancestor at node n1. Manceau & Lambert (2019) pose 3
        conventions, 1. to use a nearest age convention 2. to use a oldest age
        convention. 3. to use the timing of the mutation event as divergence
        node (usually used in UNTB models). For "nearest" and "oldest", the
        definition found in Manceau & Lambert (2019) informs us "The first two
        possibilities consist in relying on a time of divergence between
        individuals of the newly derived species and individuals of the
        ancestral, mother, species". So to find this node, we have to compute
        distances pairwise for every individuals of each species u and v.
        In this list of ages, min() is the nearest phylogenetic node and max()
        is the oldest phylogenetic node. In this example,
        mutation : 
                               ┌──── red
                            ┌──┤
                            │  └──── grey
                            └─────── yellow
                       50     25    0 BP
        nearest :
                              ┌───── red
                         ┌────┤
                         │    └───── grey
                         └────────── yellow
                       50     25    0 BP
        oldest : 
                  ┌───────────────── grey
                  │
                 ─┼───────────────── red
                  │
                  └───────────────── yellow
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
                'spmodel must be either "loose", "lacy", "genealogy" or "paraphyletic" string')
    if not isinstance(force_ultrametric, bool):
        raise ValueError('force_ultrametric must be a boolean')
    if not isinstance(debug, bool):
        raise ValueError('debug must be a boolean')
    if seed is not None and not isinstance(seed, int):
        raise ValueError('seed must be an integer')
    if seed is not None:
        np.random.seed(seed) # Initialize RNG vector
    if age not in ["mutation", "nearest", "oldest"]:
        raise ValueError("age must be either 'mutation', 'nearest' or 'oldest'")

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
        
        # For that mutation event, we convert its position of occurrence alog the branch
        # in age before present. This is for "mutation" age-convention.
        
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
    # The label "grey" still exists at the present-day because of ancestral retention, and gave rise to "green", "blue" and "purple" label
    # The label "blue" label deriving from "grey" label is also a parent label for "red" and "yellow" labels
    # "blue" label is not found in its descendants but is part of their evolutionnary history.
    # We have to count for all deriving labels and not only ones on the leaves : they are analoguous in a way to
    # internal lineages of a phylogenetic branch.
    #
    # This example holds a precious insight : a label that is part of the evolutionnary history isn't necesseraly visible
    # at the present-day. When trying to compute oldest and nearest coalescence age for the transition between
    # "grey" --> "blue" labels we need to take all individuals carrying derived labels from "blue", which are "red" and "yellow".
    # To find the nearest/oldest coalescence time between "grey" and "blue" we have to compare pairwise the min and max
    # coalescence times of all "yellow-red" individuals and all "black individuals".
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
    # parent_leaves = [purple_ind1]

            
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
            # Context : 
            # In ETE3 one simple pairwise query LCA(u,v) costs O(h) time where h is the topological
            # height of the genealogy.
            # The problem is that ecophylo has multiple mutations µ that change the leaves identities.
            # e is one readable mutation on leaves.
            # Let's have the mutation e, splitting a node between a derived D_e side and a parent
            # P_e side so that D_e = 2 leaves and P_e = 2 leaves :
            #
            #            R
            #          /   \* mutation e
            #         A     B
            #        / \   / \
            #       W  X  Y_e Z_e
            #
            # Then if Q_e = 2 x 2 = 4 ; for every e mutations readable on leaves we have :
            #                        Q = ∑_e(​∣D_e​∣ x ​∣P_e​∣)
            # For all Q pairwise comparisons it costs :
            #   - O(Qh)
            #     = O(Q log(n)) for a balanced binary tree
            #     = O(Qn) in the worst case ; upper bound
            #
            #
            # Bender & Farach-Colton show that a LCA (Least Common Ancestor) query on a tree 
            # can be reduced to a RMQ (Range Minimum Query) on the Euler_L array.
            # We can say LCA(u, v) = Euler_E[RMQ_Euler_L(Euler_R[u], Euler_R[v])].
            #
            # Their Lemma 1 states precisely that if RMQ has preprocessing/query
            # complexity <f(n), g(n)>, then LCA on an n-node tree has complexity:
            #
            #             <f(2*n - 1) + O(n), g(2*n - 1) + O(1)>
            #
            # The goal of this algorithm is to preprocess a rooted tree so that repeated LCA queries
            # can subsequently be answered in constant O(1) time and so for all pairwise comparisons
            # it would cost O(Q) query time.
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
            # Euler_L between Euler_R[u] and Euler_R[v]. For one pair (u, v), the number of Euler_L
            # positions is ​∣Euler_R[u] - Euler_R[v]​∣ + 1.
            # Therefore for all Q pairwise comparisons the number of Eleur_L positions is
            # ∑_e ∑_(u in D_e) ∑_(v in P_e) (|Euler_R[u] - Euler_R[v]| + 1)
            #
            # Since Euler_L contains 2n-1 positions, one RMQ interval can contain O(n) positions
            # Therefore O(Qn) in the worst case for all Q comparisons. With the difference of the
            # ETE3 approach moving vertically in the tree and RMQ naïve moving horizontically in the
            # Euler's arrays. Both algorithm are different but can show similar computation time.
            # Needed demonstration : maybe TBD if I've got time, if I'm really motivated or if I can't
            # sleep - written by Théo on August the 19th at 2:21am.
            # Especially upper bounded time that we should focus on as ecophylo tries to allow
            # big computation capacities.
            #
            # We now need a data structure able to answer queries without scanning the entire
            # interval RMQ_Euler_L(Euler_R[u], Euler_R[v]) that answers LCA(u, v), the proposed
            # optimisations are
            #       1. a Sparse Table processed in O(n x log(n)) time and allowing every query to be
            #          processed in O(1) time so all the algorithm costs <O(n x log(n), O(1)> for one scan
            #          and <O(n x log(n), O(Q)> for all the pairwise comparison between derived and parent side.
            #       2. a ±1-RMQ array : our L array scanned by RMQ is already a ±1-RMQ array because
            #          all possibles relations movements are toward parent/children (+1 or -1 level).
            #          That special L is called A. 
            # Section 4 then exploits the special ±1 property of the L-array to reduce preprocessing
            # to O(n) and keeping a query time to O(1). Therefore, scanning all pairwise comparisons
            # cost O(n) + O(Q) between derived and parent side. Because they showed that LCA queries
            # can be reduced to RMQ queries but where Euler-L is a ±1-RMQ array, we can apply this
            # next optimisations : 
            # Bender & Farach-Colton partition the ±1-RMQ into small blocks of size approx
            # floor(log2(euler_size)/2) and max(1, ...) prevents a block of size 0,
            # and ceil(euler_size / b_size) is the max number of blocks to completely cover
            # Euler_L. Ceil() allows the last block to be shorter if needed.
            #
            #
            
            euler_size = len(euler_L) # Number of positions, for n nodes : 2n-1 positions
            b_size = max(1, (math.floor(math.log2(euler_size) / 2))) 
            n_blocks = math.ceil(euler_size / b_size)
            
            A_prime = [] # An array storing minimum level values contained in each block, A_prime[block]
            B = [] # position of the minimum inside blocks, B[block]
            
            for block in range(n_blocks):
                block_start = block * b_size # convert block number (because range()) in a global start position in Euler_L.
                block_end = min(block_start + b_size, euler_size) # global end position (min() for the final block can't exceed Euler_L)
                block_length = block_end - block_start
                local_min_pos = 0 # initiale local position in the block
                for local_pos in range(1, block_length):
                    if (euler_L[block_start + local_pos] < euler_L[block_start + local_min_pos]): 
                        local_min_pos = local_pos # continue to iterate through the block until we find a minimum value and store its position
                A_prime.append(euler_L[block_start + local_min_pos]) # After scanning the block, store its minimum level in A_prime.
                B.append(local_min_pos) # After scanning the block, store the position where the minimum occurs.
            
            # The Sparse Table M[i][j] stores the position of the minimum in A_prime over the interval starting at i and having length 2**j
            # M[i][j] = argmin(A_prime[i: i+2**j])
            # Lets take A_prime = [7, 4, 6, 2, 8, 3, 5, 1], len() is 8
            # M[i][0] stores the argmin over 2**0 = 1 value, M[i][3] stores the argmin over 2**3 = 8 values (total length)
            # 1/8 value.s interval starting at i. Further code will make sure that only M[0][3] exists because an 8 value interval
            # (total length) can only starts from the first position as i.
            # That's why we are separating the table in columns of length log2(n)+1
            #
            # Why ? Because this allows a query to be O(1) time and not O(n) anymore.
            # positions:          0  1  2  3  4  5  6  7
            # A_prime:           [7, 4, 6, 2, 8, 3, 5, 1]
            #                           |---------------|
            #                              query [2, 7]
            # The query contains 7-2+1 = 6 values
            # We choose k = floor(log2(6)) = 2 ; Therefore 2**k = 4
            # We use two already precomputed intervals of length 4 :
            #       first interval  = positions [2, 5] = [6, 2, 8, 3]
            #       second interval = positions [4, 7] = [8, 3, 5, 1]
            # For the Sparse Table to returns these :
            #       first = M[2][2]
            #       second = M[4][2]
            # followed by : min(A_prime[first], A_prime[second]) returning the position corresponding to the smaller value.
            # So one RMQ does not scan its 6 values anymore : it uses two table lookups and one comparison, giving O(1) query time.
            #
            # A EXPLIQUER PLUS TARD:
            # Section 4 applies the Sparse Table to A_prime rather than directly
            # to the complete +/-1 input array A.
            #
            # The paper partitions A into blocks of size log(n)/2 and defines
            # A_prime with one minimum per block. Therefore A_prime contains
            # 2*n/log(n) entries.
            #
            # The paper then states that running the Sparse Table algorithm on
            # A_prime takes O(n) preprocessing time.
            
            M_levels = math.floor(math.log2(len(A_prime))) + 1 # Number of powers 2**j required to represent every possible A_prime interval length
            M = [[None] * M_levels for _ in range(len(A_prime))] # Allocate one Sparse-Table row per A_prime position, and one column per 2**j interval size
            for i in range(len(A_prime)): # Initialise the first column
                M[i][0] = i # j=0 is an interval of 1 which the minimum is necessarily at position i
            for j in range (1, M_levels): # Now we construct the following columns recursively from the previous one (dynamic code)
                interval_length = 2**j
                half_length = 2**(j - 1) 
                # All interval_length starting at i must remain completely inside A_prime
                # If len(A_prime) then the last valid starting point for EVERY interval_length is len(A_prime) - interval_length
                # so 'i' should range from 0 to (len(A_prime) - interval_length) + 1 (range exludes upper bound)
                # Now in this for loop we're doing dynamic programming
                # For the column j, the previous column j-1 already stores the minimum position of the left half of j which begins at i and has half_length (2**j-1)
                # The right half begins at half_length position after i and M already knows its minimum because it has length 2**(j-1)
                for i in range(len(A_prime) - interval_length + 1):
                    left_min = M[i][j - 1]
                    right_min = M[i + half_length][j - 1]
                    if A_prime[left_min] <= A_prime[right_min]: # Now we compare tha A_prime values of the two previous columns to store the position of the minimum of this new column, that is a glued version of the two previous columns of half_length.
                        M[i][j] = left_min
                    else:
                        M[i][j] = right_min
            
            # The Sparse-Table M can now answer RMQ queries over complete blocks through A_prime. But an arbitrary query does not
            # necessarily begin at the beginning of a block or finish at the end of a block.
            # For example : the query RMQ(i,j)
            #       [ 0 1 2 3 2 ] [ 3 4 3 2 1 ] [ 2 3 4 3 2 ] = Euler_L separated in blocks
            #            |---------------------------|
            #            i                           j
            # We can query the middle block with A_prime and M but not the beginning and ending of the query.
            # Observation 3 states that adding or subtracting the same constant from every element of an array does not change its RMQ positions.
            #
            # For example : 
            #       original_block   = [5, 6, 7, 6, 5]
            #       normalised_block = [0, 1, 2, 1, 0]
            # Both arrays have the same positions for every possible subinterval, we can therefore represent each block only by its ±1 movements
            # +1 is represented by 1 and -1 is represented by 0.
            # Lemma 4:
            # There are O(sqrt(n)) kinds of normalized blocks.
            #
            # Section 4 uses blocks containing log(n)/2 values.
            # Because adjacent values differ by +1 or -1, a normalized block
            # is completely specified by the movements between consecutive
            # values, i.e. by a +/-1 vector of length:
            #
            #                   (1/2 * log(n)) - 1
            #
            # The number of such vectors is:
            #
            #                   2**((1/2 * log(n)) - 1)
            #
            # which the paper states is O(sqrt(n)).
            
            
            
            micro_tables = {} # store one complete in-block RMQ table for every possible normalised ±1 block type.
        
            for block_length in range(1, b_size + 1): # Consider every possible block length; the final Euler_L block may be shorter then b_size
                n_signatures = 2**(block_length - 1)
                for signature in range(n_signatures):
                    normalised = [0]
                    for step in range(block_length - 1): # Reconstruct each of the k-1 movements encoded by this signature
                        shift = block_length - 2 - step # Number of binary (log2) positions remaining to the right of the movement we currently want to read
                        bit = (signature // (2**shift)) % 2 # Extract this binary digit
                        if bit == 1: # By convention, binary value 1 represents a +1 Euler movement
                            normalised.append(normalised[-1] + 1)
                        else: # By convention, binary value 0 represents a -1 Euler movement
                            normalised.append(normalised[-1] - 1)
                    
                    # Allocate a square table for this normalised block
                    #
                    # table[left][right] will contain de LOCAL POSITION of the minimum between positions left and right, included.
                    
                    table = [
                        [None] * block_length
                        for _ in range(block_length)
                    ]

                    for left in range(block_length):

                        min_pos = left
                        table[left][left] = left

                        for right in range(
                            left + 1,
                            block_length
                        ):

                            if normalised[right] < normalised[min_pos]:
                                min_pos = right

                            table[left][right] = min_pos

                    micro_tables[
                        (block_length, signature)
                    ] = table

            block_types = []

            for block in range(n_blocks):

                block_start = block * b_size
                block_end = min(
                    block_start + b_size,
                    euler_size
                )

                block_length = block_end - block_start

                # Explicit Observation 3 normalization:
                # subtract the first Level of the block from every Level.
                normalised_block = [
                    euler_L[pos] - euler_L[block_start]
                    for pos in range(
                        block_start,
                        block_end
                    )
                ]

                signature = 0

                for local_pos in range(
                    1,
                    block_length
                ):

                    signature = signature * 2

                    movement = (
                        normalised_block[local_pos]
                        - normalised_block[local_pos - 1]
                    )

                    if movement == 1:

                        # +1 is encoded by bit 1.
                        signature = signature + 1

                    elif movement == -1:

                        # -1 is represented by bit 0.
                        # The left shift already inserted this zero bit.
                        pass

                    else:

                        # This should be impossible if the Euler Tour was built
                        # correctly: two consecutive Euler levels must differ
                        # exactly by +1 or -1.
                        raise RuntimeError(
                            "Euler_L does not satisfy the +/-1 property."
                        )

                block_types.append(
                    (block_length, signature)
                )

            all_labels = {1}
            for row in mutation_table:
                all_labels.add(row["label"])
                all_labels.add(row["parent_label"])
            descendant_labels = {label: {label} for label in all_labels}
            for row in reversed(mutation_table):
                descendant_labels[row["parent_label"]].update(descendant_labels[row["label"]])
            label_to_leaves = {}
            for leaf in tree.iter_leaves():
                label_to_leaves.setdefault(leaf.sp, []).append(leaf)
            for row in mutation_table:
                derived_label = row["label"]
                parent_label = row["parent_label"]
                derived_side_labels = descendant_labels[derived_label]
                parent_side_labels = (descendant_labels[parent_label] - derived_side_labels)
                derived_leaves = []
                for label in derived_side_labels:
                    derived_leaves.extend(label_to_leaves.get(label, []))
                parent_leaves = []
                for label in parent_side_labels:
                    parent_leaves.extend(label_to_leaves.get(label, []))
                if not derived_leaves or not parent_leaves:
                    row["nearest_coalescence_age"] = None
                    row["oldest_coalescence_age"] = None
                    continue
                nearest_age = None
                oldest_age = None
                for derived_leaf in derived_leaves:
                    left_first = euler_R[id(derived_leaf)]
                    for parent_leaf in parent_leaves:
                        right_first = euler_R[id(parent_leaf)]
                        
                        if left_first <= right_first:
                            left=left_first
                            right = right_first
                        else:
                            left = right_first
                            right = left_first
                        left_block = left // b_size
                        right_block = right // b_size
                        if left_block == right_block:  # Both RMQ endpoints are inside the same small block.
                            local_left = left - left_block * b_size
                            local_right = right - left_block * b_size
                            block_length, signature = block_types[left_block]
                            local_min_pos = micro_tables[(block_length, signature)][local_left][local_right]
                            minimum_position = (left_block * b_size + local_min_pos)
                        else:
                            # --------------------------------------------------------------------------------------------
                            # Section 4 - First value:
                            # minimum from left forward to the end of its block.
                            # --------------------------------------------------------------------------------------------

                            left_block_length, left_signature = block_types[
                                left_block
                            ]  # Retrieve the normalized +/-1 type of the block containing left.


                            left_local = (
                                left - left_block * b_size
                            )  # Convert the GLOBAL left endpoint into its LOCAL position inside this block.


                            left_local_min = micro_tables[
                                (left_block_length, left_signature)
                            ][left_local][left_block_length - 1]  # In-block RMQ from left to the LAST local position of its block.


                            left_pos = (
                                left_block * b_size + left_local_min
                            )  # Convert this LOCAL minimum position back into its GLOBAL Euler_L position.
                            # --------------------------------------------------------------------------------------------
                            # Section 4 - Third value:
                            # minimum from the beginning of right's block to right.
                            # --------------------------------------------------------------------------------------------

                            right_block_length, right_signature = block_types[
                                right_block
                            ]  # Retrieve the normalized +/-1 type of the block containing right.


                            right_local = (
                                right - right_block * b_size
                            )  # Convert the GLOBAL right endpoint into its LOCAL position inside this block.


                            right_local_min = micro_tables[
                                (right_block_length, right_signature)
                            ][0][right_local]  # In-block RMQ from the FIRST local position of the block up to right.


                            right_pos = (
                                right_block * b_size + right_local_min
                            )  # Convert this LOCAL minimum position back into its GLOBAL Euler_L position.
                            # First compare the minima returned by the two boundary blocks.
                            # best_pos stores the GLOBAL Euler_L position of the smallest
                            # Level found so far.

                            if euler_L[left_pos] <= euler_L[right_pos]:

                                best_pos = left_pos

                            else:

                                best_pos = right_pos
                            # --------------------------------------------------------------------------------------------
                            # Section 4 - Second value:
                            # minimum of all COMPLETE blocks between left's block and right's block.
                            # --------------------------------------------------------------------------------------------

                            middle_left = left_block + 1  # First complete block strictly after the block containing left.

                            middle_right = right_block - 1  # Last complete block strictly before the block containing right.
                            if middle_left <= middle_right:  # Enter only when at least one complete intermediate block exists.

                                middle_length = (
                                    middle_right - middle_left + 1
                                )  # Number of complete blocks represented in this A_prime query.
                                k = math.floor(
                                    math.log2(middle_length)
                                )  # Largest k such that 2**k fits inside the queried A_prime interval.
                                first = M[
                                    middle_left
                                ][k]  # Argmin in the power-of-two interval beginning at middle_left.
                                second_start = (
                                    middle_right - 2**k + 1
                                )  # Starting position of a second 2**k interval ending exactly at middle_right.


                                second = M[
                                    second_start
                                ][k]  # Argmin in this right-aligned power-of-two interval.
                                if A_prime[first] <= A_prime[second]:  # Compare the two candidate minimum VALUES in A_prime.

                                    middle_block = first  # The minimum of the complete middle range comes from this block.

                                else:  # The second candidate contains the smaller block minimum.

                                    middle_block = second  # Keep the corresponding A_prime position.
                                middle_pos = (
                                    middle_block * b_size
                                    + B[middle_block]
                                )  # Recover the GLOBAL Euler_L position from which A_prime[middle_block] originated.
                                if euler_L[middle_pos] < euler_L[best_pos]:  # Compare the complete-middle minimum with the best boundary minimum.

                                    best_pos = middle_pos  # The middle range contains the global RMQ minimum.
                            minimum_position = best_pos  # The minimum among the three Section-4 values is the answer to RMQ(left, right).   
                        mrca = euler_E[
                            minimum_position
                        ]  # Convert the RMQ position back into the genealogy LCA through the Euler array E.
                        mrca_age = node_age[
                            id(mrca)
                        ]  # Retrieve the already-computed age of this MRCA without another genealogy traversal.
                        if nearest_age is None or mrca_age < nearest_age:  # Check whether this pair gives the smallest MRCA age encountered so far.

                            nearest_age = mrca_age  # Update the running minimum corresponding to the nearest convention.


                        if oldest_age is None or mrca_age > oldest_age:  # Check whether this pair gives the largest MRCA age encountered so far.

                            oldest_age = mrca_age  # Update the running maximum corresponding to the oldest convention.
                row["nearest_coalescence_age"] = nearest_age  # Store the minimum MRCA age found over the complete derived_leaves x parent_leaves Cartesian product.

                row["oldest_coalescence_age"] = oldest_age  # Store the maximum MRCA age found over exactly the same individual pairs.


        # ========================================================================================================
        # Reconstruct the paraphyletic label-tree
        # ========================================================================================================
        #
        # At this point mutation_table contains the divergence age that must be
        # used for every historical mutation event:
        #
        #       mutation -> mutation_coalescence_age
        #       nearest  -> nearest_coalescence_age
        #       oldest   -> oldest_coalescence_age
        #
        # age_column, defined above, selects the appropriate one.
        #
        # We now reconstruct a phylogenetic label-tree from PRESENT-DAY labels.
        # One phylogenetic leaf is created for each distinct label still carried
        # by at least one present-day individual.

        lineage = {}  # Map each currently reconstructed historical label to its phylogenetic subtree.


        for leaf in tree.iter_leaves():  # Visit every present-day individual in the original genealogy.

            label = leaf.sp  # Final mutation label carried by this present-day individual.


            if label not in lineage:  # First present-day individual encountered with this label.

                popInd = [0] * (ndeme + 1)  # Create the abundance vector used by EcoPhylo to record individuals among demes.

                popInd[leaf.deme] = 1  # Count this first individual in its deme.


                new_leaf = type(tree)()  # Create a new ETE TreeNode that will represent this complete present-day label.

                new_leaf.add_features(
                    sp=label,  # Preserve the mutation/species label represented by this phylogenetic tip.
                    name="sp" + str(label),  # Name the reconstructed tip from its label.
                    popInd=popInd,  # Store present-day abundance among demes.
                    mergedInd=str(leaf.name),  # Record the first genealogy individual represented by this tip.
                    _age=0.0  # Every reconstructed tip exists at present, therefore its age is zero.
                )

                lineage[label] = new_leaf  # This new phylogenetic tip is now the reconstructed subtree associated with this label.
            else:  # A phylogenetic tip already exists for this present-day label.

                lineage[label].popInd[leaf.deme] += 1  # Add this individual to the abundance of the corresponding deme.

                lineage[label].mergedInd += " " + str(leaf.name)  # Record this additional genealogy individual inside the same phylogenetic tip.
        internal_index = 0  # Counter used only to give unique names n0, n1, ... to reconstructed internal phylogenetic nodes.
        ordered_events = sorted(
            mutation_table,
            key=lambda row: (
                row[age_column] is None,
                row["mutation_coalescence_age"]
                if row[age_column] is None
                else row[age_column]
            )
        )
        for event in ordered_events:  # Reconstruct historical label transitions from younger divergence ages toward older ones.

            parent_label = event["parent_label"]  # Label existing before the mutation.

            derived_label = event["label"]  # Unique label created by this mutation.         
            
            if event[age_column] is None:  # No pairwise divergence node can be positioned for this historical transition.

                if derived_label in lineage and parent_label not in lineage:  # A surviving derived subtree exists but no separate parent subtree exists.

                    lineage[parent_label] = lineage[derived_label]  # Propagate the same surviving subtree to the historical parent label without creating a node.

                continue  # This event does not create a phylogenetic divergence node.

            event_age = float(event[age_column])  # Divergence age selected by the user's mutation/nearest/oldest convention.

            if derived_label not in lineage:

                continue
            
            derived_tree = lineage[derived_label]  # Reconstructed phylogenetic subtree descending from the derived label.

            if parent_label not in lineage:  # No independently represented parent-side subtree exists yet.

                lineage[parent_label] = derived_tree  # Propagate the derived subtree upward as the current representation of the parent lineage.

                continue  # No branching node is required until another represented lineage has to be joined.

            parent_tree = lineage[parent_label]

            if (
                not parent_tree.is_leaf()
                and abs(parent_tree._age - event_age) < 1e-12
            ):

                replacement_node = parent_tree
            else:

                replacement_node = type(tree)()  # Create a new internal phylogenetic node for this divergence.

                replacement_node.name = "n" + str(internal_index)  # Give it a unique internal-node name.

                internal_index += 1  # Reserve the next identifier for the next newly created internal node.

                replacement_node.add_features(
                    sp=None,  # Internal reconstructed nodes do not represent a present-day species label.
                    _age=event_age  # Store the absolute divergence age represented by this node.
                )
                
                parent_tree.dist = (
                    event_age - parent_tree._age
                )  # Branch length equals the difference between the older parent age and the younger child-subtree age.

                replacement_node.add_child(
                    parent_tree
                )  # Attach the previously reconstructed parent-side subtree below this new divergence node.
                
            derived_tree.dist = (
                event_age - derived_tree._age
            )  # Set the branch length connecting the derived subtree to the divergence age.


            replacement_node.add_child(
                derived_tree
            )  # Attach the complete derived subtree to the reconstructed divergence node.
            
            lineage[parent_label] = replacement_node  # The complete merged subtree now becomes the reconstructed lineage associated with the historical parent label.
            
        tree = lineage[1]  # Label 1 is the ancestral label, so its reconstructed subtree is the complete paraphyletic phylogeny.

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
        age = age)
    
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