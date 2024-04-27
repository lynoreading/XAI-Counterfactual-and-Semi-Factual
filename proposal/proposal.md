## iML Project Proposal

#### roup name: weare3, member: Yu Li, Yifan Wang, (Hans Olbrich)

\
For the group project in iML, we would like to work on a publication on generating counterfactual and semi-factual explanations [KK20]. We want to reimplement the generation of counterfactuals and semi-factuals with the new
method PlausIble Exceptionality-based Contrastive Explanations (PIECE), for
which the author provided code in a GitHub repository https://github.com/Eoin
Kenny/AAAI-2021?tab=readme-ov-file. The tasks (TODO) involved include setting up the environment, explain the code and conducting
training with dataset which author used (possible to try training on
a more complex dataset), and to generate the final results and diagram. \
\
In their research paper, the authors conduct comparative analyses between
PIECE and other broadly applicable methodologies suitable for color datasets,
such as the MNIST dataset. The methods used for comparison include Min Edit,
C-Min-Edit, CEM, and Proto-CF. For the counterfactual part, metrics MC
Mean, MC STD, NN-Dist, IM1 and R-Sub were used, and for the semi-factual
part, the author used three diagrams to show that the semi-factuals generated
by PIECE are superior to several other methods, such as L1 Pixel Distance
from Test Image to Explanation Image. The tasks (TODO) involved include a deeper understanding and explanation of these six methods
(PIECE and the other five methods used for comparison), identifying
their main differences. A more in-depth study of these metrics is
required to explain their principles and the properties of the models
they reflect.\
\
While the original paper used the MNIST digital dataset, other nondigital datasets, such as FASHION MNIST, can be tried to investigate the effect of the method on the images of objects. find out
more on (https://www.simonwenkel.com/lists/datasets/list-of-mnistlike-datasets.html). \
\
(TODO) For extensions, explore the advantages and disadvantages
between semi-factual and counterfactual. In what situations is it
more appropriate to use semi-factual, and in what situations is it
more suitable to use counterfactual? Are there suitable metrics to
measure this? (if possible, we would like to receive some suggestions
for extensions part, in situations where the task difficulty is not too
high. What can we do in this part)\
\
References
[KK20] Eoin M. Kenny and Mark T. Keane. On Generating Plausible Counterfactual and Semi-Factual Explanations for Deep Learning. 2020. arXiv:
2009.06399 [cs.LG]
