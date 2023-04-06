CSCI 5461 Spring 2023 Machine Learning Prediction Challenge A & B

03/23/2023 by Xiang Zhang (zhan6668@umn.edu)

---------------------------------------------------

Video introduction of these challenges by Prof. Chad Myers
https://canvas.umn.edu/courses/355386/external_tools/24

---------------------------------------------------

Note about data sharing: These data are unpublished and are being shared for the purposes of CSCI5461 projects. Please do not post these data publicly or distribute the data beyond CSCI5461 students without Prof. Myers's permission. Thank you!

---------------------------------------------------

This folder includes all the files and their descriptions for Challenge A and Challenge B.

Both challenges share this input file: 

Challenge_GIN_release_profile_17804library_181query.txt
- Genetic interaction profiles for 17,804 library genes and 181 query genes
- The names of the all the query genes were masked
- The names of the first 5,000 library genes (221 query genes + 4,779 randomly selected) were masked
- The 181 query genes have presence on the library side as well (e.g., gene2 on the query side represents the same gene as gene2 on the library side)

---------------------------------------------------

Challenge A:

ChallengeA_release_GO_BP_20_500_17804library_1106terms.txt
- Annotation of 1,106 Gene Ontology Biological Process terms for 17,804 library genes
- A 1 in a given position of this matrix means that the gene is annotated with the corresponding column’s GO term, a -1 means that the gene is not annotated with the corresponding column’s GO term, and a 0 means that we would like you to make predictions for that gene 
- For the first 5,000 library genes (randomly selected), their GO BP annotations have been held back for us to evaluate the performance of your predictions, and thus are all 0s. 
- Please train and tune your algorithm excluding these first 5,000 genes, and submit your prediction for their GO BP annotations based on your optimized model (i.e., submit a 5,000 x 1,106 matrix).

---------------------------------------------------

Challenge B:

ChallengeB_release_hold_out_40_query_genes.txt
- We withheld these 40 genes from the query side for validating your predictions, which means you will only see them on the library side of the GIN file above and have to submit your prediction for their genetic interaction scores across the library as if they were screened as queries (i.e., submit a 17,804 x 40 matrix).

---------------------------------------------------

If you have any question regarding the data, please feel free to contact: 
- Xiang Zhang (TA, zhan6668@umn.edu)
- Mehrad Hajiaghabozorgi (TA, hajia018@umn.edu)
- Dr. Chad Myers (Instructor, chadm@umn.edu)
