# -*- coding: utf-8 -*-
"""
Created on Fri Nov 6 13:20:00 2020

@author : Maxime Jaunatre <maxime.jaunatre@yahoo.fr>
@author : Elizabeth Bathelemy <barthelemy.elizabeth@gmail.com>
@author : Théo Driancourt <imevolutionmodding@gmail.com>

Functions : 
    toPhylo
    ubranch_mutation

"""
# TODO : more info in help toPhylo

import numpy as np

def toPhylo(tree, mu, tau = 0, spmodel = "phenotypic", 
            force_ultrametric = True, seed = None, debug = False):
    """
    Merge branches of genealogy following speciation model of the user choice 
    after sprinkling mutation events over the branches of simulated genealogies
    depending on branch lengths. 
    
    Mutation events are sprinkled over the branches of simulated genealogies 
    depending on branch lengths, so that the number of mutations over a branch 
    follows a Poisson distribution with parameter 𝜇·𝐵 where 𝜇 is the point 
    mutation rate and 𝐵 is the length of the branch. 
    
    The descendants stemming from a branch with at least one mutation define
    a genetically distinct clade. These clades can be paraphyletic because of
    ancestral retention. Either the user choose to accept paraphyletic gene
    histories ('spmodel = phenotypic'), or to impose monophyly. For every
    gene tree exist a species partition satisfying either A or B. These
    differences in species tree can be compared to the taxonomist opposition
    between splitter and lumper.
    
    Parameters
    ----------
    tree : TreeNode (ete3 class)
        A tree representing the genealogy of simulated individuals.
    mu : float
        point mutation rate, must be comprised between between 0 and 1. 
    tau = 0 : float
        The minimum number of generations monophyletic lineages have to be 
        seperated for to be considered distinct species
        
        
    spmodel = {"genealogy", "loose", "lacy", "phenotypic"}
        default = "phenotypic" : string

        - "genealogy"
        The complete genealogy is retained after mutation sprinkling.
        Cannot be coerced in a species tree.
        
        - "loose"
        The complete genealogy is collapsed in a set of monophyletic clades
        according to the "loose species partition" from Manceau, Lambert 2018.
        The loose species partition aims to respect heterotypy between species,
        individuals in different species are genetically different for each
        species cluster. This is the finest partition of the present-day
        individuals as it usually needs to merge different clades : individuals
        can have different labels within a species. This is equivalent to
        lumpers.
        
        - "lacy"
        The complete genealogy is collapsed in a set of monophyletic clades
        according to the "lacy species partition" from Manceau, Lambert 2018.
        The lacy species partition aims to respect homotypy within species,
        individuals within the same species are genetically identical for each
        species cluster. This is the coarsest partition of the present-day
        individuals as it usually needs to XXXXX : individuals with the same
        labels can be in different species. This is equivalent to splitters.
        
        - "phenotypic"
        The complete genealogy is collapsed in a set of monophyletic lineages
        as a "label-tree" rather than a species tree. This is equivalent to
        Hubbell's UNTB and associated research (Jabot & Chave 2009). See more
        in Manceau, Lambert 2018 but note what we call "XXXXX" is analogous to
        what they call "phenotypic", implying phenotypes.

    force_ultrametric = True : bool
        Whether or note to force phylogenetic tree ultrametry 
    seed = None : int
        None by default, set the seed for mutation random events.
    debug = False : bool
        change the return function, returns a tuple containing the phylogeny,
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
    # Idiot proof
    if tree.__class__.__name__ != 'TreeNode' :
        raise ValueError('tree must have a class TreeNode')
    if mu < 0 or mu > 1 or not isinstance(mu, (int,float)):
        raise ValueError('mu must be a float between 0 and 1')
    if not spmodel in ['loose', 'lacy', 'genealogy', 'phenotypic']:
        raise ValueError(spmodel+' is not a correct model. '+
                'spmodel must be either "loose", "lacy", "genealogy" or "phenotypic" string')
    if not isinstance(force_ultrametric, bool):
        raise ValueError('force_ultrametric must be a boolean')
    if not isinstance(debug, bool):
        raise ValueError('debug must be a boolean')
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
    
    sp_origin = {1: tree}
    tree_height = max(tree.get_distance(leaf) for leaf in tree.iter_leaves())
    mutation_table = [] 
    
    # mutation model on branches
    for node in tree.traverse("preorder"): # traverse les noeuds
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
            
        is_leaf = node.is_leaf()
        
        # Inherit sp label from parent, if None, it means that we are at the
        # root, we set it to 1 because preorder begins with the root. So we 
        # don't allow mutation above the root (ubranch_mutation would return
        # False anyway) and we don't label it as an internal node because
        # node.up.sp doesn't exist for the root
        if node.up is None:
            node.add_features(sp=1)
            continue
        else:
            node.add_features(sp=node.up.sp)
        
        umut, mut_from_parent = ubranch_mutation(node = node, mu = mu, tau = tau)
        
        if not umut:
            continue
        
        # umut is True : new label overwritting
        
        parent_label = node.sp # get parent sp from node before overwritting it
        spID += 1
        node.sp = spID
        node.mut = "*" # for debugging when printing the tree
        sp_origin[spID] = node
        
        # Mutation age information
        
        parent_name = node.up.name
        child_name = node.name
        
        parent_depth = tree.get_distance(node.up)  
        mutation_depth = parent_depth + mut_from_parent
       
        mutation_coalescence_age = tree_height - mutation_depth
        mutation_id = str(parent_name) + "->" + str(child_name)
        
        node.add_features(mutation_id = mutation_id, mut_from_parent = mut_from_parent, mutation_coalescence_age = mutation_coalescence_age)
        
        mutation_table.append({
            "mutation_id": mutation_id,
            "is_terminal": is_leaf,
            "label": spID,
            "parent_label": parent_label,
            "mutation_coalescence_age": mutation_coalescence_age, # Lambert scenario c
            "nearest_coalescence_age": None, # Lambert scenario a
            "oldest_coalescence_age": None # Lambert scenario b
            })

    # Labelling is complete : we can compute Lambert scenarios a & b here
    # 
    # For each mutation event, a parent label gives rise to a derived label
    # There's multiple outcomes possible : 
    # - A derived label survives up until present
    # - A derived label becomes a parent label by giving rise to a new derived label and exists in the present-day
    # - A derived label becomes a parent label by giving rise to multiple new derived labels and only none of its descendants
    #   in the present-day carry the label, existing only as an internal label.
    #
    # Example, marked by a star all speciation events :
    #                         ┌───────*purple─── purple_ind1
    #                  ┌──────┤
    #                  │      └───────────────── grey_ind1
    #                  │           
    #                  │      
    #                  │                  ┌───── red_ind1
    #          grey  ──┤             ┌*red┤
    #                  │             │    └───── red_ind2
    #                  │      ┌─*blue┤
    #                  │      │      │        ┌─ yellow_ind1
    #                  │      │      └*yellow─┤
    #                  │      │               └─ yellow_ind2
    #                  └──────┤
    #                         │         ┌*green─ green_ind1
    #                         │      ┌──┤
    #                         │      │  └─────── grey_ind2
    #                         └──────┤
    #                                │  ┌─────── grey_ind3
    #                                └──┤
    #                                   └─────── grey_ind4
    #
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
    #
    # Finir plus tard une fois le code complété




    parent_of = {} # Dictionary storing the direct parent of each derived label
    all_labels = set([1]) # Store all labels ever created; label 1 is the ancestral root label; set() stores unique elements

    for row in mutation_table :
        derived_label = row["label"] # blue
        parent_label = row["parent_label"] # grey

        parent_of[derived_label] = parent_label # Store the transition parent --> derived as parent_of[blue] = grey
        
        all_labels.add(derived_label) # Add the derived label to all historical labels
        all_labels.add(parent_label) # Add the parent label to all historical labels
    
    descendant_labels = {}
    
    for label in all_labels :
        descendant_labels[label] = set([label]) # if label = grey then descendant_labels[grey] = {grey} for now
                                                # if label = blue then descendant_labels[blue] = {blue} for now
        
    for row in reversed(mutation_table) :
        derived_label = row["label"]
        parent_label = row["parent_label"]
        
        descendant_labels[parent_label].update(descendant_labels[derived_label])
        # This adds every descendant of derived_label to the descendant set of parent_label
        # For example, in this tree we have these transitions : 
        #     grey -> purple
        #     grey -> blue
        #     blue -> yellow
        #     blue -> red
        #     grey -> green
        # mutation_table is filled in preorder. Therefore, reversed(mutation_table) walks from derived labels back towards
        # older parents labels, just like postorder, in other words if you have a row B --> C then you'll never have
        # a row C --> X above it, always below it.
        #
        # Recursively :
        # row "grey -> green" | descendant_labels[grey].update(descendant_labels[green]) = {grey, green}
        # row "blue -> red"   | descendant_labels[blue].update(descendant_labels[red]) = {blue, red}
        # row "blue -> yellow"| descendant_labels[blue].update(descendant_labels[yellow]) = {blue, red, yellow}
        # row "grey -> blue"  | descendant_labels[grey].update(descendant_labels[blue]) = {grey, green, blue, red, yellow}
        # row "grey -> purple"| descendant_labels[grey].update(descendant_labels[purple]) = {grey, green, blue, red, yellow, purple}

    label_to_leaves = {}
        # Dictionary storing the present-day leaves carrying each final label.
        # Example :
        #    label_to_leaves[red] = [red_ind1, red_ind2]
        #    label_to_leaves[grey] = [grey_ind1, grey_ind2, grey_ind3, grey_ind4]
        #    label_to_leaves[blue] = does not exist because no present-day individual is carrying the blue label
    
    for leaf in tree.iter_leaves() :
        if leaf.sp not in label_to_leaves : # if never encountered yet leaf.sp
            label_to_leaves[leaf.sp] = [] # create its key in the dictionary
        label_to_leaves[leaf.sp].append(leaf) # append the leaf (ind) to its species/label key
                                              # so for example label_to_leaves[red] = [red_ind1, red_ind2]

    # All this was mainly to compute "descendant_labels"

    for row in mutation_table :
        derived_label = row["label"]
        parent_label = row["parent_label"]
        
        derived_side_labels = descendant_labels[derived_label]
        parent_side_labels = descendant_labels[parent_label] - descendant_labels[derived_label]
        # Mutation_table:
        #     grey -> purple
        #     grey -> blue
        #     blue -> yellow
        #     blue -> red
        #     grey -> green
        #
        # Recursively : 
        # row "grey -> purple"
        #       - derived_side_labels = descendant_labels[purple] = {purple}
        #       - parent_side_labels = descendant_labels[grey] - descendant_labels[purple]
        #         = {grey, green, blue, red, yellow, purple} - {purple} = {grey, green, blue, red, yellow}
        # row "grey -> blue"
        #       - derived_side_labels = descendant_labels[blue] = {blue, red, yellow}
        #       - parent_side_labels = descendant_labels[grey] - descendant_labels[blue]
        #         = {grey, green, blue, red, yellow, purple} - {blue, red, yellow,} = {grey, green, purple}
        #
        # Now we may compare pairwise all individuals of parent_side to all individuals derived_side to find the min() and max()
        # coalescent age.
        
        derived_leaves = []
        for label in derived_side_labels : 
            derived_leaves.extend(label_to_leaves.get(label, []))
            # We have ids : all derived labels, we have have dictionary converting labels to present-day leaves is they exist
            # we can transform all derived_side_labels to derived_leaves by getting from the dictionnary, if "None" then []
            #
            # Let's take the "grey -> blue" example : 
            # derived_side_labels = {blue, red, yellow}
            # label = blue then extend derived_leaves by [] : there are no present-day blue leaves
            # label = red then extend derived_leaves by [red_ind1, red_ind2]
            # label = yellow then extend derived_leaves by [yellow_ind1, yellow_ind2]
            # So to get coalescent_age of the transition "grey -> blue" you have to analyse
            # derived_leaves = [red_ind1, red_ind2, yellow_ind1, yellow_ind2].
        
        parent_leaves = []
        for label in parent_side_labels : 
            parent_leaves.extend(label_to_leaves.get(label, []))
            # Same but for parent_side
            # for "grey -> blue" :
            # parent_leaves = [purple_ind1, grey_ind1, green_ind1, grey_ind2, grey_ind3, grey_ind4].
            
            
        if len(derived_leaves) == 0 or len(parent_leaves) == 0 :
            row["nearest_coalescence_age"] = None
            row["oldest_coalescence_age"] = None
            print('Something went wrong : a derived_leaves or parent_leaves list is empty')
            continue
            # if one side is empty, its an error keep everything None and continue
            # this shouldn't happen, if a label creates a derived label, then they both
            # exist on the branches of the genealogy : which always lead to living
            # individual, even if their label is overwriten.
            
        coalescence_ages = []
        # Store all pairwise MRCA ages between derived_leaves and parent_leaves
        # Example for "grey -> blue" :
        #   derived_leaves = [red_ind1, red_ind2, yellow_ind1, yellow_ind2]
        #   parent_leaves = [purple_ind1, grey_ind1, green_ind1, grey_ind2, grey_ind3, grey_ind4]
        # We shall compute all pairwise combinations between these two lists
        # min() is scenario a and max() is scenario b
        
        for derived_leaf in derived_leaves:
            for parent_leaf in parent_leaves :
                mrca = tree.get_common_ancestor(derived_leaf, parent_leaf) # Get the node
                mrca_age = tree_height - tree.get_distance(mrca) # Get the age of that node ; root_to_present - root_to_MRCAnode = age before present
                coalescence_ages.append(mrca_age)
        
        row["nearest_coalescence_age"] = min(coalescence_ages)
        # Lambert scenario a, nearest = the most recent pairwise coalescence between the derived side and the parent side
        
        row["oldest_coalescence_age"] = max(coalescence_ages)
        # Lambert scenario b, oldest = the oldest pairwise coalescence between the derived side and the parent side




            
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
                    
                                    
                
    if spmodel == "phenotypic" :
        for leaf in tree.iter_leaves():
            popInd = [0] * (ndeme + 1)
            popInd[leaf.deme] = 1
            leaf.popInd = popInd
        
    # if spmodel == "paraphyletic" :
    
    
    



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
        mutation_table=mutation_table)
    
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
