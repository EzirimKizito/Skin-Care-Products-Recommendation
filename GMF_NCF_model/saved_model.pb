��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
�
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.15.02v2.15.0-0-g6887368d6d48��
v
countVarHandleOp*
_output_shapes
: *

debug_namecount/*
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
v
totalVarHandleOp*
_output_shapes
: *

debug_nametotal/*
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
|
count_1VarHandleOp*
_output_shapes
: *

debug_name
count_1/*
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
|
total_1VarHandleOp*
_output_shapes
: *

debug_name
total_1/*
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
|
count_2VarHandleOp*
_output_shapes
: *

debug_name
count_2/*
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
|
total_2VarHandleOp*
_output_shapes
: *

debug_name
total_2/*
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
|
count_3VarHandleOp*
_output_shapes
: *

debug_name
count_3/*
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
|
total_3VarHandleOp*
_output_shapes
: *

debug_name
total_3/*
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
�
Adam/v/dense_9/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/v/dense_9/bias/*
dtype0*
shape:*$
shared_nameAdam/v/dense_9/bias
w
'Adam/v/dense_9/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_9/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_9/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/m/dense_9/bias/*
dtype0*
shape:*$
shared_nameAdam/m/dense_9/bias
w
'Adam/m/dense_9/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_9/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_9/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/dense_9/kernel/*
dtype0*
shape
:`*&
shared_nameAdam/v/dense_9/kernel

)Adam/v/dense_9/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_9/kernel*
_output_shapes

:`*
dtype0
�
Adam/m/dense_9/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/dense_9/kernel/*
dtype0*
shape
:`*&
shared_nameAdam/m/dense_9/kernel

)Adam/m/dense_9/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_9/kernel*
_output_shapes

:`*
dtype0
�
Adam/v/dense_8/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/v/dense_8/bias/*
dtype0*
shape: *$
shared_nameAdam/v/dense_8/bias
w
'Adam/v/dense_8/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_8/bias*
_output_shapes
: *
dtype0
�
Adam/m/dense_8/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/m/dense_8/bias/*
dtype0*
shape: *$
shared_nameAdam/m/dense_8/bias
w
'Adam/m/dense_8/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_8/bias*
_output_shapes
: *
dtype0
�
Adam/v/dense_8/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/dense_8/kernel/*
dtype0*
shape
:@ *&
shared_nameAdam/v/dense_8/kernel

)Adam/v/dense_8/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_8/kernel*
_output_shapes

:@ *
dtype0
�
Adam/m/dense_8/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/dense_8/kernel/*
dtype0*
shape
:@ *&
shared_nameAdam/m/dense_8/kernel

)Adam/m/dense_8/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_8/kernel*
_output_shapes

:@ *
dtype0
�
Adam/v/dense_5/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/v/dense_5/bias/*
dtype0*
shape:@*$
shared_nameAdam/v/dense_5/bias
w
'Adam/v/dense_5/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_5/bias*
_output_shapes
:@*
dtype0
�
Adam/m/dense_5/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/m/dense_5/bias/*
dtype0*
shape:@*$
shared_nameAdam/m/dense_5/bias
w
'Adam/m/dense_5/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_5/bias*
_output_shapes
:@*
dtype0
�
Adam/v/dense_5/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/dense_5/kernel/*
dtype0*
shape
:2@*&
shared_nameAdam/v/dense_5/kernel

)Adam/v/dense_5/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_5/kernel*
_output_shapes

:2@*
dtype0
�
Adam/m/dense_5/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/dense_5/kernel/*
dtype0*
shape
:2@*&
shared_nameAdam/m/dense_5/kernel

)Adam/m/dense_5/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_5/kernel*
_output_shapes

:2@*
dtype0
�
Adam/v/dense_7/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/v/dense_7/bias/*
dtype0*
shape:@*$
shared_nameAdam/v/dense_7/bias
w
'Adam/v/dense_7/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_7/bias*
_output_shapes
:@*
dtype0
�
Adam/m/dense_7/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/m/dense_7/bias/*
dtype0*
shape:@*$
shared_nameAdam/m/dense_7/bias
w
'Adam/m/dense_7/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_7/bias*
_output_shapes
:@*
dtype0
�
Adam/v/dense_7/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/dense_7/kernel/*
dtype0*
shape:	�@*&
shared_nameAdam/v/dense_7/kernel
�
)Adam/v/dense_7/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_7/kernel*
_output_shapes
:	�@*
dtype0
�
Adam/m/dense_7/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/dense_7/kernel/*
dtype0*
shape:	�@*&
shared_nameAdam/m/dense_7/kernel
�
)Adam/m/dense_7/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_7/kernel*
_output_shapes
:	�@*
dtype0
�
Adam/v/dense_6/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/v/dense_6/bias/*
dtype0*
shape:�*$
shared_nameAdam/v/dense_6/bias
x
'Adam/v/dense_6/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_6/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_6/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/m/dense_6/bias/*
dtype0*
shape:�*$
shared_nameAdam/m/dense_6/bias
x
'Adam/m/dense_6/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_6/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_6/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/dense_6/kernel/*
dtype0*
shape:
��*&
shared_nameAdam/v/dense_6/kernel
�
)Adam/v/dense_6/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_6/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense_6/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/dense_6/kernel/*
dtype0*
shape:
��*&
shared_nameAdam/m/dense_6/kernel
�
)Adam/m/dense_6/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_6/kernel* 
_output_shapes
:
��*
dtype0
�
!Adam/v/brand_embedding/embeddingsVarHandleOp*
_output_shapes
: *2

debug_name$"Adam/v/brand_embedding/embeddings/*
dtype0*
shape
:N2*2
shared_name#!Adam/v/brand_embedding/embeddings
�
5Adam/v/brand_embedding/embeddings/Read/ReadVariableOpReadVariableOp!Adam/v/brand_embedding/embeddings*
_output_shapes

:N2*
dtype0
�
!Adam/m/brand_embedding/embeddingsVarHandleOp*
_output_shapes
: *2

debug_name$"Adam/m/brand_embedding/embeddings/*
dtype0*
shape
:N2*2
shared_name#!Adam/m/brand_embedding/embeddings
�
5Adam/m/brand_embedding/embeddings/Read/ReadVariableOpReadVariableOp!Adam/m/brand_embedding/embeddings*
_output_shapes

:N2*
dtype0
�
%Adam/v/skin_type_embedding/embeddingsVarHandleOp*
_output_shapes
: *6

debug_name(&Adam/v/skin_type_embedding/embeddings/*
dtype0*
shape
:
*6
shared_name'%Adam/v/skin_type_embedding/embeddings
�
9Adam/v/skin_type_embedding/embeddings/Read/ReadVariableOpReadVariableOp%Adam/v/skin_type_embedding/embeddings*
_output_shapes

:
*
dtype0
�
%Adam/m/skin_type_embedding/embeddingsVarHandleOp*
_output_shapes
: *6

debug_name(&Adam/m/skin_type_embedding/embeddings/*
dtype0*
shape
:
*6
shared_name'%Adam/m/skin_type_embedding/embeddings
�
9Adam/m/skin_type_embedding/embeddings/Read/ReadVariableOpReadVariableOp%Adam/m/skin_type_embedding/embeddings*
_output_shapes

:
*
dtype0
�
%Adam/v/skin_tone_embedding/embeddingsVarHandleOp*
_output_shapes
: *6

debug_name(&Adam/v/skin_tone_embedding/embeddings/*
dtype0*
shape
:
*6
shared_name'%Adam/v/skin_tone_embedding/embeddings
�
9Adam/v/skin_tone_embedding/embeddings/Read/ReadVariableOpReadVariableOp%Adam/v/skin_tone_embedding/embeddings*
_output_shapes

:
*
dtype0
�
%Adam/m/skin_tone_embedding/embeddingsVarHandleOp*
_output_shapes
: *6

debug_name(&Adam/m/skin_tone_embedding/embeddings/*
dtype0*
shape
:
*6
shared_name'%Adam/m/skin_tone_embedding/embeddings
�
9Adam/m/skin_tone_embedding/embeddings/Read/ReadVariableOpReadVariableOp%Adam/m/skin_tone_embedding/embeddings*
_output_shapes

:
*
dtype0
�
(Adam/v/product_name_embedding/embeddingsVarHandleOp*
_output_shapes
: *9

debug_name+)Adam/v/product_name_embedding/embeddings/*
dtype0*
shape:	�2*9
shared_name*(Adam/v/product_name_embedding/embeddings
�
<Adam/v/product_name_embedding/embeddings/Read/ReadVariableOpReadVariableOp(Adam/v/product_name_embedding/embeddings*
_output_shapes
:	�2*
dtype0
�
(Adam/m/product_name_embedding/embeddingsVarHandleOp*
_output_shapes
: *9

debug_name+)Adam/m/product_name_embedding/embeddings/*
dtype0*
shape:	�2*9
shared_name*(Adam/m/product_name_embedding/embeddings
�
<Adam/m/product_name_embedding/embeddings/Read/ReadVariableOpReadVariableOp(Adam/m/product_name_embedding/embeddings*
_output_shapes
:	�2*
dtype0
�
 Adam/v/user_embedding/embeddingsVarHandleOp*
_output_shapes
: *1

debug_name#!Adam/v/user_embedding/embeddings/*
dtype0*
shape:	�C2*1
shared_name" Adam/v/user_embedding/embeddings
�
4Adam/v/user_embedding/embeddings/Read/ReadVariableOpReadVariableOp Adam/v/user_embedding/embeddings*
_output_shapes
:	�C2*
dtype0
�
 Adam/m/user_embedding/embeddingsVarHandleOp*
_output_shapes
: *1

debug_name#!Adam/m/user_embedding/embeddings/*
dtype0*
shape:	�C2*1
shared_name" Adam/m/user_embedding/embeddings
�
4Adam/m/user_embedding/embeddings/Read/ReadVariableOpReadVariableOp Adam/m/user_embedding/embeddings*
_output_shapes
:	�C2*
dtype0
�
learning_rateVarHandleOp*
_output_shapes
: *

debug_namelearning_rate/*
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
�
	iterationVarHandleOp*
_output_shapes
: *

debug_name
iteration/*
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
dense_9/biasVarHandleOp*
_output_shapes
: *

debug_namedense_9/bias/*
dtype0*
shape:*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:*
dtype0
�
dense_9/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_9/kernel/*
dtype0*
shape
:`*
shared_namedense_9/kernel
q
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes

:`*
dtype0
�
dense_8/biasVarHandleOp*
_output_shapes
: *

debug_namedense_8/bias/*
dtype0*
shape: *
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
: *
dtype0
�
dense_8/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_8/kernel/*
dtype0*
shape
:@ *
shared_namedense_8/kernel
q
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes

:@ *
dtype0
�
dense_5/biasVarHandleOp*
_output_shapes
: *

debug_namedense_5/bias/*
dtype0*
shape:@*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:@*
dtype0
�
dense_5/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_5/kernel/*
dtype0*
shape
:2@*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:2@*
dtype0
�
dense_7/biasVarHandleOp*
_output_shapes
: *

debug_namedense_7/bias/*
dtype0*
shape:@*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:@*
dtype0
�
dense_7/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_7/kernel/*
dtype0*
shape:	�@*
shared_namedense_7/kernel
r
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes
:	�@*
dtype0
�
dense_6/biasVarHandleOp*
_output_shapes
: *

debug_namedense_6/bias/*
dtype0*
shape:�*
shared_namedense_6/bias
j
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes	
:�*
dtype0
�
dense_6/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_6/kernel/*
dtype0*
shape:
��*
shared_namedense_6/kernel
s
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel* 
_output_shapes
:
��*
dtype0
�
brand_embedding/embeddingsVarHandleOp*
_output_shapes
: *+

debug_namebrand_embedding/embeddings/*
dtype0*
shape
:N2*+
shared_namebrand_embedding/embeddings
�
.brand_embedding/embeddings/Read/ReadVariableOpReadVariableOpbrand_embedding/embeddings*
_output_shapes

:N2*
dtype0
�
skin_type_embedding/embeddingsVarHandleOp*
_output_shapes
: */

debug_name!skin_type_embedding/embeddings/*
dtype0*
shape
:
*/
shared_name skin_type_embedding/embeddings
�
2skin_type_embedding/embeddings/Read/ReadVariableOpReadVariableOpskin_type_embedding/embeddings*
_output_shapes

:
*
dtype0
�
skin_tone_embedding/embeddingsVarHandleOp*
_output_shapes
: */

debug_name!skin_tone_embedding/embeddings/*
dtype0*
shape
:
*/
shared_name skin_tone_embedding/embeddings
�
2skin_tone_embedding/embeddings/Read/ReadVariableOpReadVariableOpskin_tone_embedding/embeddings*
_output_shapes

:
*
dtype0
�
!product_name_embedding/embeddingsVarHandleOp*
_output_shapes
: *2

debug_name$"product_name_embedding/embeddings/*
dtype0*
shape:	�2*2
shared_name#!product_name_embedding/embeddings
�
5product_name_embedding/embeddings/Read/ReadVariableOpReadVariableOp!product_name_embedding/embeddings*
_output_shapes
:	�2*
dtype0
�
user_embedding/embeddingsVarHandleOp*
_output_shapes
: **

debug_nameuser_embedding/embeddings/*
dtype0*
shape:	�C2**
shared_nameuser_embedding/embeddings
�
-user_embedding/embeddings/Read/ReadVariableOpReadVariableOpuser_embedding/embeddings*
_output_shapes
:	�C2*
dtype0
~
serving_default_brand_inputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
!serving_default_ingredients_inputPlaceholder*(
_output_shapes
:����������*
dtype0*
shape:����������
~
serving_default_loves_inputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
~
serving_default_price_inputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
"serving_default_product_name_inputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������

serving_default_rating_inputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
serving_default_review_inputPlaceholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�
serving_default_skin_tone_inputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
serving_default_skin_type_inputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
}
serving_default_user_inputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_brand_input!serving_default_ingredients_inputserving_default_loves_inputserving_default_price_input"serving_default_product_name_inputserving_default_rating_inputserving_default_review_inputserving_default_skin_tone_inputserving_default_skin_type_inputserving_default_user_inputbrand_embedding/embeddingsskin_type_embedding/embeddingsskin_tone_embedding/embeddings!product_name_embedding/embeddingsuser_embedding/embeddingsdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_5/kerneldense_5/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/bias*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*1
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_452257

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-0
layer-5
layer_with_weights-1
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer_with_weights-5
layer-21
layer-22
layer_with_weights-6
layer-23
layer_with_weights-7
layer-24
layer_with_weights-8
layer-25
layer-26
layer_with_weights-9
layer-27
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
#_default_save_signature
$	optimizer
%
signatures*
* 
* 
* 
* 
* 
�
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
,
embeddings*
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses
3
embeddings*
�
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses
:
embeddings*
�
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses
A
embeddings*
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
H
embeddings*
�
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses* 
�
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses* 
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses* 
�
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses* 
�
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses* 
* 
* 
* 
* 
* 
�
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses* 
�
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses

skernel
tbias*
�
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses* 
�
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
z
,0
31
:2
A3
H4
s5
t6
�7
�8
�9
�10
�11
�12
�13
�14*
z
,0
31
:2
A3
H4
s5
t6
�7
�8
�9
�10
�11
�12
�13
�14*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
#_default_save_signature
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla*

�serving_default* 

,0*

,0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
mg
VARIABLE_VALUEuser_embedding/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

30*

30*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
uo
VARIABLE_VALUE!product_name_embedding/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

:0*

:0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
rl
VARIABLE_VALUEskin_tone_embedding/embeddings:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

A0*

A0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
rl
VARIABLE_VALUEskin_type_embedding/embeddings:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

H0*

H0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
nh
VARIABLE_VALUEbrand_embedding/embeddings:layer_with_weights-4/embeddings/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

s0
t1*

s0
t1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_6/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_6/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_7/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_7/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_5/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_5/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_8/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_8/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_9/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_9/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27*
$
�0
�1
�2
�3*
* 
* 
* 
* 
* 
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
ke
VARIABLE_VALUE Adam/m/user_embedding/embeddings1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/v/user_embedding/embeddings1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE(Adam/m/product_name_embedding/embeddings1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE(Adam/v/product_name_embedding/embeddings1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE%Adam/m/skin_tone_embedding/embeddings1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE%Adam/v/skin_tone_embedding/embeddings1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE%Adam/m/skin_type_embedding/embeddings1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE%Adam/v/skin_type_embedding/embeddings1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE!Adam/m/brand_embedding/embeddings1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/brand_embedding/embeddings2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_6/kernel2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_6/kernel2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_6/bias2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_6/bias2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_7/kernel2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_7/kernel2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_7/bias2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_7/bias2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_5/kernel2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_5/kernel2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_5/bias2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_5/bias2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_8/kernel2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_8/kernel2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_8/bias2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_8/bias2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_9/kernel2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_9/kernel2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_9/bias2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_9/bias2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_34keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_34keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameuser_embedding/embeddings!product_name_embedding/embeddingsskin_tone_embedding/embeddingsskin_type_embedding/embeddingsbrand_embedding/embeddingsdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_5/kerneldense_5/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/bias	iterationlearning_rate Adam/m/user_embedding/embeddings Adam/v/user_embedding/embeddings(Adam/m/product_name_embedding/embeddings(Adam/v/product_name_embedding/embeddings%Adam/m/skin_tone_embedding/embeddings%Adam/v/skin_tone_embedding/embeddings%Adam/m/skin_type_embedding/embeddings%Adam/v/skin_type_embedding/embeddings!Adam/m/brand_embedding/embeddings!Adam/v/brand_embedding/embeddingsAdam/m/dense_6/kernelAdam/v/dense_6/kernelAdam/m/dense_6/biasAdam/v/dense_6/biasAdam/m/dense_7/kernelAdam/v/dense_7/kernelAdam/m/dense_7/biasAdam/v/dense_7/biasAdam/m/dense_5/kernelAdam/v/dense_5/kernelAdam/m/dense_5/biasAdam/v/dense_5/biasAdam/m/dense_8/kernelAdam/v/dense_8/kernelAdam/m/dense_8/biasAdam/v/dense_8/biasAdam/m/dense_9/kernelAdam/v/dense_9/kernelAdam/m/dense_9/biasAdam/v/dense_9/biastotal_3count_3total_2count_2total_1count_1totalcountConst*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_452906
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameuser_embedding/embeddings!product_name_embedding/embeddingsskin_tone_embedding/embeddingsskin_type_embedding/embeddingsbrand_embedding/embeddingsdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_5/kerneldense_5/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/bias	iterationlearning_rate Adam/m/user_embedding/embeddings Adam/v/user_embedding/embeddings(Adam/m/product_name_embedding/embeddings(Adam/v/product_name_embedding/embeddings%Adam/m/skin_tone_embedding/embeddings%Adam/v/skin_tone_embedding/embeddings%Adam/m/skin_type_embedding/embeddings%Adam/v/skin_type_embedding/embeddings!Adam/m/brand_embedding/embeddings!Adam/v/brand_embedding/embeddingsAdam/m/dense_6/kernelAdam/v/dense_6/kernelAdam/m/dense_6/biasAdam/v/dense_6/biasAdam/m/dense_7/kernelAdam/v/dense_7/kernelAdam/m/dense_7/biasAdam/v/dense_7/biasAdam/m/dense_5/kernelAdam/v/dense_5/kernelAdam/m/dense_5/biasAdam/v/dense_5/biasAdam/m/dense_8/kernelAdam/v/dense_8/kernelAdam/m/dense_8/biasAdam/v/dense_8/biasAdam/m/dense_9/kernelAdam/v/dense_9/kernelAdam/m/dense_9/biasAdam/v/dense_9/biastotal_3count_3total_2count_2total_1count_1totalcount*C
Tin<
:28*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_453080ٮ
�
�
O__inference_skin_tone_embedding_layer_call_and_return_conditional_losses_452305

inputs)
embedding_lookup_452300:

identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:����������
embedding_lookupResourceGatherembedding_lookup_452300Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/452300*+
_output_shapes
:���������
*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:���������
u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:���������
5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:&"
 
_user_specified_name452300
�
p
F__inference_multiply_1_layer_call_and_return_conditional_losses_451872

inputs
inputs_1
identityN
mulMulinputsinputs_1*
T0*'
_output_shapes
:���������2O
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:���������2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������2:���������2:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�

�
C__inference_dense_7_layer_call_and_return_conditional_losses_452473

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
0__inference_brand_embedding_layer_call_fn_452328

inputs
unknown:N2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_brand_embedding_layer_call_and_return_conditional_losses_451732s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������2<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:&"
 
_user_specified_name452324
�
�
(__inference_dense_6_layer_call_fn_452430

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_451845p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:&"
 
_user_specified_name452424:&"
 
_user_specified_name452426
�

�
C__inference_dense_8_layer_call_and_return_conditional_losses_452513

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
K__inference_brand_embedding_layer_call_and_return_conditional_losses_452337

inputs)
embedding_lookup_452332:N2
identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:����������
embedding_lookupResourceGatherembedding_lookup_452332Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/452332*+
_output_shapes
:���������2*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:���������2u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:���������25
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:&"
 
_user_specified_name452332
�
�
(__inference_model_1_layer_call_fn_452079

user_input
skin_tone_input
skin_type_input
brand_input
product_name_input
price_input
rating_input
loves_input
review_input
ingredients_input
unknown:N2
	unknown_0:

	unknown_1:

	unknown_2:	�2
	unknown_3:	�C2
	unknown_4:
��
	unknown_5:	�
	unknown_6:	�@
	unknown_7:@
	unknown_8:2@
	unknown_9:@

unknown_10:@ 

unknown_11: 

unknown_12:`

unknown_13:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
user_inputskin_tone_inputskin_type_inputbrand_inputproduct_name_inputprice_inputrating_inputloves_inputreview_inputingredients_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*1
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_451991o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:����������:����������: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:���������
$
_user_specified_name
user_input:XT
'
_output_shapes
:���������
)
_user_specified_nameskin_tone_input:XT
'
_output_shapes
:���������
)
_user_specified_nameskin_type_input:TP
'
_output_shapes
:���������
%
_user_specified_namebrand_input:[W
'
_output_shapes
:���������
,
_user_specified_nameproduct_name_input:TP
'
_output_shapes
:���������
%
_user_specified_nameprice_input:UQ
'
_output_shapes
:���������
&
_user_specified_namerating_input:TP
'
_output_shapes
:���������
%
_user_specified_nameloves_input:VR
(
_output_shapes
:����������
&
_user_specified_namereview_input:[	W
(
_output_shapes
:����������
+
_user_specified_nameingredients_input:&
"
 
_user_specified_name452047:&"
 
_user_specified_name452049:&"
 
_user_specified_name452051:&"
 
_user_specified_name452053:&"
 
_user_specified_name452055:&"
 
_user_specified_name452057:&"
 
_user_specified_name452059:&"
 
_user_specified_name452061:&"
 
_user_specified_name452063:&"
 
_user_specified_name452065:&"
 
_user_specified_name452067:&"
 
_user_specified_name452069:&"
 
_user_specified_name452071:&"
 
_user_specified_name452073:&"
 
_user_specified_name452075
�
�
$__inference_signature_wrapper_452257
brand_input
ingredients_input
loves_input
price_input
product_name_input
rating_input
review_input
skin_tone_input
skin_type_input

user_input
unknown:N2
	unknown_0:

	unknown_1:

	unknown_2:	�2
	unknown_3:	�C2
	unknown_4:
��
	unknown_5:	�
	unknown_6:	�@
	unknown_7:@
	unknown_8:2@
	unknown_9:@

unknown_10:@ 

unknown_11: 

unknown_12:`

unknown_13:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
user_inputskin_tone_inputskin_type_inputbrand_inputproduct_name_inputprice_inputrating_inputloves_inputreview_inputingredients_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*1
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_451712o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:����������:���������:���������:���������:���������:����������:���������:���������:���������: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:���������
%
_user_specified_namebrand_input:[W
(
_output_shapes
:����������
+
_user_specified_nameingredients_input:TP
'
_output_shapes
:���������
%
_user_specified_nameloves_input:TP
'
_output_shapes
:���������
%
_user_specified_nameprice_input:[W
'
_output_shapes
:���������
,
_user_specified_nameproduct_name_input:UQ
'
_output_shapes
:���������
&
_user_specified_namerating_input:VR
(
_output_shapes
:����������
&
_user_specified_namereview_input:XT
'
_output_shapes
:���������
)
_user_specified_nameskin_tone_input:XT
'
_output_shapes
:���������
)
_user_specified_nameskin_type_input:S	O
'
_output_shapes
:���������
$
_user_specified_name
user_input:&
"
 
_user_specified_name452225:&"
 
_user_specified_name452227:&"
 
_user_specified_name452229:&"
 
_user_specified_name452231:&"
 
_user_specified_name452233:&"
 
_user_specified_name452235:&"
 
_user_specified_name452237:&"
 
_user_specified_name452239:&"
 
_user_specified_name452241:&"
 
_user_specified_name452243:&"
 
_user_specified_name452245:&"
 
_user_specified_name452247:&"
 
_user_specified_name452249:&"
 
_user_specified_name452251:&"
 
_user_specified_name452253
�
�
R__inference_product_name_embedding_layer_call_and_return_conditional_losses_452289

inputs*
embedding_lookup_452284:	�2
identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:����������
embedding_lookupResourceGatherembedding_lookup_452284Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/452284*+
_output_shapes
:���������2*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:���������2u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:���������25
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:&"
 
_user_specified_name452284
�
�
(__inference_dense_5_layer_call_fn_452482

inputs
unknown:2@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_451884o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs:&"
 
_user_specified_name452476:&"
 
_user_specified_name452478
�
a
E__inference_flatten_7_layer_call_and_return_conditional_losses_451803

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����
   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������
X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
:S O
+
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
J__inference_user_embedding_layer_call_and_return_conditional_losses_452273

inputs*
embedding_lookup_452268:	�C2
identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:����������
embedding_lookupResourceGatherembedding_lookup_452268Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/452268*+
_output_shapes
:���������2*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:���������2u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:���������25
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:&"
 
_user_specified_name452268
�
F
*__inference_flatten_9_layer_call_fn_452353

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_flatten_9_layer_call_and_return_conditional_losses_451817`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2:S O
+
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
(__inference_dense_9_layer_call_fn_452535

inputs
unknown:`
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_451923o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������`: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs:&"
 
_user_specified_name452529:&"
 
_user_specified_name452531
�
a
E__inference_flatten_5_layer_call_and_return_conditional_losses_452348

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����2   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������2X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2:S O
+
_output_shapes
:���������2
 
_user_specified_nameinputs
�Y
�	
C__inference_model_1_layer_call_and_return_conditional_losses_451930

user_input
skin_tone_input
skin_type_input
brand_input
product_name_input
price_input
rating_input
loves_input
review_input
ingredients_input(
brand_embedding_451733:N2,
skin_type_embedding_451745:
,
skin_tone_embedding_451757:
0
product_name_embedding_451769:	�2(
user_embedding_451781:	�C2"
dense_6_451846:
��
dense_6_451848:	�!
dense_7_451862:	�@
dense_7_451864:@ 
dense_5_451885:2@
dense_5_451887:@ 
dense_8_451901:@ 
dense_8_451903:  
dense_9_451924:`
dense_9_451926:
identity��'brand_embedding/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�.product_name_embedding/StatefulPartitionedCall�+skin_tone_embedding/StatefulPartitionedCall�+skin_type_embedding/StatefulPartitionedCall�&user_embedding/StatefulPartitionedCall�
'brand_embedding/StatefulPartitionedCallStatefulPartitionedCallbrand_inputbrand_embedding_451733*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_brand_embedding_layer_call_and_return_conditional_losses_451732�
+skin_type_embedding/StatefulPartitionedCallStatefulPartitionedCallskin_type_inputskin_type_embedding_451745*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_skin_type_embedding_layer_call_and_return_conditional_losses_451744�
+skin_tone_embedding/StatefulPartitionedCallStatefulPartitionedCallskin_tone_inputskin_tone_embedding_451757*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_skin_tone_embedding_layer_call_and_return_conditional_losses_451756�
.product_name_embedding/StatefulPartitionedCallStatefulPartitionedCallproduct_name_inputproduct_name_embedding_451769*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_product_name_embedding_layer_call_and_return_conditional_losses_451768�
&user_embedding/StatefulPartitionedCallStatefulPartitionedCall
user_inputuser_embedding_451781*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_user_embedding_layer_call_and_return_conditional_losses_451780�
flatten_5/PartitionedCallPartitionedCall/user_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_451789�
flatten_6/PartitionedCallPartitionedCall4skin_tone_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_flatten_6_layer_call_and_return_conditional_losses_451796�
flatten_7/PartitionedCallPartitionedCall4skin_type_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_flatten_7_layer_call_and_return_conditional_losses_451803�
flatten_8/PartitionedCallPartitionedCall0brand_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_flatten_8_layer_call_and_return_conditional_losses_451810�
flatten_9/PartitionedCallPartitionedCall7product_name_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_flatten_9_layer_call_and_return_conditional_losses_451817�
concatenate_2/PartitionedCallPartitionedCall"flatten_5/PartitionedCall:output:0"flatten_6/PartitionedCall:output:0"flatten_7/PartitionedCall:output:0"flatten_8/PartitionedCall:output:0"flatten_9/PartitionedCall:output:0price_inputrating_inputloves_inputreview_inputingredients_input*
Tin
2
*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_concatenate_2_layer_call_and_return_conditional_losses_451833�
dense_6/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_6_451846dense_6_451848*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_451845�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_451862dense_7_451864*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_451861�
multiply_1/PartitionedCallPartitionedCall"flatten_5/PartitionedCall:output:0"flatten_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_multiply_1_layer_call_and_return_conditional_losses_451872�
dense_5/StatefulPartitionedCallStatefulPartitionedCall#multiply_1/PartitionedCall:output:0dense_5_451885dense_5_451887*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_451884�
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_451901dense_8_451903*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_451900�
concatenate_3/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_451912�
dense_9/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0dense_9_451924dense_9_451926*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_451923w
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp(^brand_embedding/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall/^product_name_embedding/StatefulPartitionedCall,^skin_tone_embedding/StatefulPartitionedCall,^skin_type_embedding/StatefulPartitionedCall'^user_embedding/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:����������:����������: : : : : : : : : : : : : : : 2R
'brand_embedding/StatefulPartitionedCall'brand_embedding/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2`
.product_name_embedding/StatefulPartitionedCall.product_name_embedding/StatefulPartitionedCall2Z
+skin_tone_embedding/StatefulPartitionedCall+skin_tone_embedding/StatefulPartitionedCall2Z
+skin_type_embedding/StatefulPartitionedCall+skin_type_embedding/StatefulPartitionedCall2P
&user_embedding/StatefulPartitionedCall&user_embedding/StatefulPartitionedCall:S O
'
_output_shapes
:���������
$
_user_specified_name
user_input:XT
'
_output_shapes
:���������
)
_user_specified_nameskin_tone_input:XT
'
_output_shapes
:���������
)
_user_specified_nameskin_type_input:TP
'
_output_shapes
:���������
%
_user_specified_namebrand_input:[W
'
_output_shapes
:���������
,
_user_specified_nameproduct_name_input:TP
'
_output_shapes
:���������
%
_user_specified_nameprice_input:UQ
'
_output_shapes
:���������
&
_user_specified_namerating_input:TP
'
_output_shapes
:���������
%
_user_specified_nameloves_input:VR
(
_output_shapes
:����������
&
_user_specified_namereview_input:[	W
(
_output_shapes
:����������
+
_user_specified_nameingredients_input:&
"
 
_user_specified_name451733:&"
 
_user_specified_name451745:&"
 
_user_specified_name451757:&"
 
_user_specified_name451769:&"
 
_user_specified_name451781:&"
 
_user_specified_name451846:&"
 
_user_specified_name451848:&"
 
_user_specified_name451862:&"
 
_user_specified_name451864:&"
 
_user_specified_name451885:&"
 
_user_specified_name451887:&"
 
_user_specified_name451901:&"
 
_user_specified_name451903:&"
 
_user_specified_name451924:&"
 
_user_specified_name451926
�	
�
C__inference_dense_9_layer_call_and_return_conditional_losses_452545

inputs0
matmul_readvariableop_resource:`-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
(__inference_model_1_layer_call_fn_452035

user_input
skin_tone_input
skin_type_input
brand_input
product_name_input
price_input
rating_input
loves_input
review_input
ingredients_input
unknown:N2
	unknown_0:

	unknown_1:

	unknown_2:	�2
	unknown_3:	�C2
	unknown_4:
��
	unknown_5:	�
	unknown_6:	�@
	unknown_7:@
	unknown_8:2@
	unknown_9:@

unknown_10:@ 

unknown_11: 

unknown_12:`

unknown_13:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
user_inputskin_tone_inputskin_type_inputbrand_inputproduct_name_inputprice_inputrating_inputloves_inputreview_inputingredients_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*1
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_451930o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:����������:����������: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:���������
$
_user_specified_name
user_input:XT
'
_output_shapes
:���������
)
_user_specified_nameskin_tone_input:XT
'
_output_shapes
:���������
)
_user_specified_nameskin_type_input:TP
'
_output_shapes
:���������
%
_user_specified_namebrand_input:[W
'
_output_shapes
:���������
,
_user_specified_nameproduct_name_input:TP
'
_output_shapes
:���������
%
_user_specified_nameprice_input:UQ
'
_output_shapes
:���������
&
_user_specified_namerating_input:TP
'
_output_shapes
:���������
%
_user_specified_nameloves_input:VR
(
_output_shapes
:����������
&
_user_specified_namereview_input:[	W
(
_output_shapes
:����������
+
_user_specified_nameingredients_input:&
"
 
_user_specified_name452003:&"
 
_user_specified_name452005:&"
 
_user_specified_name452007:&"
 
_user_specified_name452009:&"
 
_user_specified_name452011:&"
 
_user_specified_name452013:&"
 
_user_specified_name452015:&"
 
_user_specified_name452017:&"
 
_user_specified_name452019:&"
 
_user_specified_name452021:&"
 
_user_specified_name452023:&"
 
_user_specified_name452025:&"
 
_user_specified_name452027:&"
 
_user_specified_name452029:&"
 
_user_specified_name452031
�

�
C__inference_dense_5_layer_call_and_return_conditional_losses_452493

inputs0
matmul_readvariableop_resource:2@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
W
+__inference_multiply_1_layer_call_fn_452447
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_multiply_1_layer_call_and_return_conditional_losses_451872`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������2:���������2:Q M
'
_output_shapes
:���������2
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:���������2
"
_user_specified_name
inputs_1
�
�
4__inference_skin_tone_embedding_layer_call_fn_452296

inputs
unknown:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_skin_tone_embedding_layer_call_and_return_conditional_losses_451756s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:&"
 
_user_specified_name452292
�

�
C__inference_dense_8_layer_call_and_return_conditional_losses_451900

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

�
C__inference_dense_6_layer_call_and_return_conditional_losses_452441

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
4__inference_skin_type_embedding_layer_call_fn_452312

inputs
unknown:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_skin_type_embedding_layer_call_and_return_conditional_losses_451744s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:&"
 
_user_specified_name452308
�
F
*__inference_flatten_8_layer_call_fn_452386

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_flatten_8_layer_call_and_return_conditional_losses_451810`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2:S O
+
_output_shapes
:���������2
 
_user_specified_nameinputs
�
a
E__inference_flatten_5_layer_call_and_return_conditional_losses_451789

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����2   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������2X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2:S O
+
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
O__inference_skin_tone_embedding_layer_call_and_return_conditional_losses_451756

inputs)
embedding_lookup_451751:

identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:����������
embedding_lookupResourceGatherembedding_lookup_451751Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/451751*+
_output_shapes
:���������
*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:���������
u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:���������
5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:&"
 
_user_specified_name451751
�
�
O__inference_skin_type_embedding_layer_call_and_return_conditional_losses_451744

inputs)
embedding_lookup_451739:

identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:����������
embedding_lookupResourceGatherembedding_lookup_451739Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/451739*+
_output_shapes
:���������
*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:���������
u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:���������
5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:&"
 
_user_specified_name451739
��
�2
__inference__traced_save_452906
file_prefixC
0read_disablecopyonread_user_embedding_embeddings:	�C2M
:read_1_disablecopyonread_product_name_embedding_embeddings:	�2I
7read_2_disablecopyonread_skin_tone_embedding_embeddings:
I
7read_3_disablecopyonread_skin_type_embedding_embeddings:
E
3read_4_disablecopyonread_brand_embedding_embeddings:N2;
'read_5_disablecopyonread_dense_6_kernel:
��4
%read_6_disablecopyonread_dense_6_bias:	�:
'read_7_disablecopyonread_dense_7_kernel:	�@3
%read_8_disablecopyonread_dense_7_bias:@9
'read_9_disablecopyonread_dense_5_kernel:2@4
&read_10_disablecopyonread_dense_5_bias:@:
(read_11_disablecopyonread_dense_8_kernel:@ 4
&read_12_disablecopyonread_dense_8_bias: :
(read_13_disablecopyonread_dense_9_kernel:`4
&read_14_disablecopyonread_dense_9_bias:-
#read_15_disablecopyonread_iteration:	 1
'read_16_disablecopyonread_learning_rate: M
:read_17_disablecopyonread_adam_m_user_embedding_embeddings:	�C2M
:read_18_disablecopyonread_adam_v_user_embedding_embeddings:	�C2U
Bread_19_disablecopyonread_adam_m_product_name_embedding_embeddings:	�2U
Bread_20_disablecopyonread_adam_v_product_name_embedding_embeddings:	�2Q
?read_21_disablecopyonread_adam_m_skin_tone_embedding_embeddings:
Q
?read_22_disablecopyonread_adam_v_skin_tone_embedding_embeddings:
Q
?read_23_disablecopyonread_adam_m_skin_type_embedding_embeddings:
Q
?read_24_disablecopyonread_adam_v_skin_type_embedding_embeddings:
M
;read_25_disablecopyonread_adam_m_brand_embedding_embeddings:N2M
;read_26_disablecopyonread_adam_v_brand_embedding_embeddings:N2C
/read_27_disablecopyonread_adam_m_dense_6_kernel:
��C
/read_28_disablecopyonread_adam_v_dense_6_kernel:
��<
-read_29_disablecopyonread_adam_m_dense_6_bias:	�<
-read_30_disablecopyonread_adam_v_dense_6_bias:	�B
/read_31_disablecopyonread_adam_m_dense_7_kernel:	�@B
/read_32_disablecopyonread_adam_v_dense_7_kernel:	�@;
-read_33_disablecopyonread_adam_m_dense_7_bias:@;
-read_34_disablecopyonread_adam_v_dense_7_bias:@A
/read_35_disablecopyonread_adam_m_dense_5_kernel:2@A
/read_36_disablecopyonread_adam_v_dense_5_kernel:2@;
-read_37_disablecopyonread_adam_m_dense_5_bias:@;
-read_38_disablecopyonread_adam_v_dense_5_bias:@A
/read_39_disablecopyonread_adam_m_dense_8_kernel:@ A
/read_40_disablecopyonread_adam_v_dense_8_kernel:@ ;
-read_41_disablecopyonread_adam_m_dense_8_bias: ;
-read_42_disablecopyonread_adam_v_dense_8_bias: A
/read_43_disablecopyonread_adam_m_dense_9_kernel:`A
/read_44_disablecopyonread_adam_v_dense_9_kernel:`;
-read_45_disablecopyonread_adam_m_dense_9_bias:;
-read_46_disablecopyonread_adam_v_dense_9_bias:+
!read_47_disablecopyonread_total_3: +
!read_48_disablecopyonread_count_3: +
!read_49_disablecopyonread_total_2: +
!read_50_disablecopyonread_count_2: +
!read_51_disablecopyonread_total_1: +
!read_52_disablecopyonread_count_1: )
read_53_disablecopyonread_total: )
read_54_disablecopyonread_count: 
savev2_const
identity_111��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
Read/DisableCopyOnReadDisableCopyOnRead0read_disablecopyonread_user_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp0read_disablecopyonread_user_embedding_embeddings^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�C2*
dtype0j
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�C2b

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:	�C2�
Read_1/DisableCopyOnReadDisableCopyOnRead:read_1_disablecopyonread_product_name_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp:read_1_disablecopyonread_product_name_embedding_embeddings^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�2*
dtype0n

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�2d

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:	�2�
Read_2/DisableCopyOnReadDisableCopyOnRead7read_2_disablecopyonread_skin_tone_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp7read_2_disablecopyonread_skin_tone_embedding_embeddings^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:
*
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:
c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:
�
Read_3/DisableCopyOnReadDisableCopyOnRead7read_3_disablecopyonread_skin_type_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp7read_3_disablecopyonread_skin_type_embedding_embeddings^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:
*
dtype0m

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:
c

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes

:
�
Read_4/DisableCopyOnReadDisableCopyOnRead3read_4_disablecopyonread_brand_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp3read_4_disablecopyonread_brand_embedding_embeddings^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:N2*
dtype0m

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:N2c

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

:N2{
Read_5/DisableCopyOnReadDisableCopyOnRead'read_5_disablecopyonread_dense_6_kernel"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp'read_5_disablecopyonread_dense_6_kernel^Read_5/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0p
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��y
Read_6/DisableCopyOnReadDisableCopyOnRead%read_6_disablecopyonread_dense_6_bias"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp%read_6_disablecopyonread_dense_6_bias^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes	
:�{
Read_7/DisableCopyOnReadDisableCopyOnRead'read_7_disablecopyonread_dense_7_kernel"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp'read_7_disablecopyonread_dense_7_kernel^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�@*
dtype0o
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�@f
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@y
Read_8/DisableCopyOnReadDisableCopyOnRead%read_8_disablecopyonread_dense_7_bias"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp%read_8_disablecopyonread_dense_7_bias^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:@{
Read_9/DisableCopyOnReadDisableCopyOnRead'read_9_disablecopyonread_dense_5_kernel"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp'read_9_disablecopyonread_dense_5_kernel^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:2@*
dtype0n
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:2@e
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes

:2@{
Read_10/DisableCopyOnReadDisableCopyOnRead&read_10_disablecopyonread_dense_5_bias"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp&read_10_disablecopyonread_dense_5_bias^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:@}
Read_11/DisableCopyOnReadDisableCopyOnRead(read_11_disablecopyonread_dense_8_kernel"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp(read_11_disablecopyonread_dense_8_kernel^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@ *
dtype0o
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@ e
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes

:@ {
Read_12/DisableCopyOnReadDisableCopyOnRead&read_12_disablecopyonread_dense_8_bias"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp&read_12_disablecopyonread_dense_8_bias^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
: }
Read_13/DisableCopyOnReadDisableCopyOnRead(read_13_disablecopyonread_dense_9_kernel"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp(read_13_disablecopyonread_dense_9_kernel^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:`*
dtype0o
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:`e
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes

:`{
Read_14/DisableCopyOnReadDisableCopyOnRead&read_14_disablecopyonread_dense_9_bias"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp&read_14_disablecopyonread_dense_9_bias^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_15/DisableCopyOnReadDisableCopyOnRead#read_15_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp#read_15_disablecopyonread_iteration^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_16/DisableCopyOnReadDisableCopyOnRead'read_16_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp'read_16_disablecopyonread_learning_rate^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_17/DisableCopyOnReadDisableCopyOnRead:read_17_disablecopyonread_adam_m_user_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp:read_17_disablecopyonread_adam_m_user_embedding_embeddings^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�C2*
dtype0p
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�C2f
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:	�C2�
Read_18/DisableCopyOnReadDisableCopyOnRead:read_18_disablecopyonread_adam_v_user_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp:read_18_disablecopyonread_adam_v_user_embedding_embeddings^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�C2*
dtype0p
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�C2f
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:	�C2�
Read_19/DisableCopyOnReadDisableCopyOnReadBread_19_disablecopyonread_adam_m_product_name_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOpBread_19_disablecopyonread_adam_m_product_name_embedding_embeddings^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�2*
dtype0p
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�2f
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:	�2�
Read_20/DisableCopyOnReadDisableCopyOnReadBread_20_disablecopyonread_adam_v_product_name_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOpBread_20_disablecopyonread_adam_v_product_name_embedding_embeddings^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�2*
dtype0p
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�2f
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:	�2�
Read_21/DisableCopyOnReadDisableCopyOnRead?read_21_disablecopyonread_adam_m_skin_tone_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp?read_21_disablecopyonread_adam_m_skin_tone_embedding_embeddings^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:
*
dtype0o
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:
e
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes

:
�
Read_22/DisableCopyOnReadDisableCopyOnRead?read_22_disablecopyonread_adam_v_skin_tone_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp?read_22_disablecopyonread_adam_v_skin_tone_embedding_embeddings^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:
*
dtype0o
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:
e
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes

:
�
Read_23/DisableCopyOnReadDisableCopyOnRead?read_23_disablecopyonread_adam_m_skin_type_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp?read_23_disablecopyonread_adam_m_skin_type_embedding_embeddings^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:
*
dtype0o
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:
e
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes

:
�
Read_24/DisableCopyOnReadDisableCopyOnRead?read_24_disablecopyonread_adam_v_skin_type_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp?read_24_disablecopyonread_adam_v_skin_type_embedding_embeddings^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:
*
dtype0o
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:
e
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes

:
�
Read_25/DisableCopyOnReadDisableCopyOnRead;read_25_disablecopyonread_adam_m_brand_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp;read_25_disablecopyonread_adam_m_brand_embedding_embeddings^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:N2*
dtype0o
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:N2e
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes

:N2�
Read_26/DisableCopyOnReadDisableCopyOnRead;read_26_disablecopyonread_adam_v_brand_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp;read_26_disablecopyonread_adam_v_brand_embedding_embeddings^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:N2*
dtype0o
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:N2e
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes

:N2�
Read_27/DisableCopyOnReadDisableCopyOnRead/read_27_disablecopyonread_adam_m_dense_6_kernel"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp/read_27_disablecopyonread_adam_m_dense_6_kernel^Read_27/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_28/DisableCopyOnReadDisableCopyOnRead/read_28_disablecopyonread_adam_v_dense_6_kernel"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp/read_28_disablecopyonread_adam_v_dense_6_kernel^Read_28/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_29/DisableCopyOnReadDisableCopyOnRead-read_29_disablecopyonread_adam_m_dense_6_bias"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp-read_29_disablecopyonread_adam_m_dense_6_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_30/DisableCopyOnReadDisableCopyOnRead-read_30_disablecopyonread_adam_v_dense_6_bias"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp-read_30_disablecopyonread_adam_v_dense_6_bias^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_31/DisableCopyOnReadDisableCopyOnRead/read_31_disablecopyonread_adam_m_dense_7_kernel"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp/read_31_disablecopyonread_adam_m_dense_7_kernel^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�@*
dtype0p
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�@f
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@�
Read_32/DisableCopyOnReadDisableCopyOnRead/read_32_disablecopyonread_adam_v_dense_7_kernel"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp/read_32_disablecopyonread_adam_v_dense_7_kernel^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�@*
dtype0p
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�@f
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@�
Read_33/DisableCopyOnReadDisableCopyOnRead-read_33_disablecopyonread_adam_m_dense_7_bias"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp-read_33_disablecopyonread_adam_m_dense_7_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_34/DisableCopyOnReadDisableCopyOnRead-read_34_disablecopyonread_adam_v_dense_7_bias"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp-read_34_disablecopyonread_adam_v_dense_7_bias^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_35/DisableCopyOnReadDisableCopyOnRead/read_35_disablecopyonread_adam_m_dense_5_kernel"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp/read_35_disablecopyonread_adam_m_dense_5_kernel^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:2@*
dtype0o
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:2@e
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes

:2@�
Read_36/DisableCopyOnReadDisableCopyOnRead/read_36_disablecopyonread_adam_v_dense_5_kernel"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp/read_36_disablecopyonread_adam_v_dense_5_kernel^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:2@*
dtype0o
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:2@e
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes

:2@�
Read_37/DisableCopyOnReadDisableCopyOnRead-read_37_disablecopyonread_adam_m_dense_5_bias"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp-read_37_disablecopyonread_adam_m_dense_5_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_38/DisableCopyOnReadDisableCopyOnRead-read_38_disablecopyonread_adam_v_dense_5_bias"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp-read_38_disablecopyonread_adam_v_dense_5_bias^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_39/DisableCopyOnReadDisableCopyOnRead/read_39_disablecopyonread_adam_m_dense_8_kernel"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp/read_39_disablecopyonread_adam_m_dense_8_kernel^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@ *
dtype0o
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@ e
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes

:@ �
Read_40/DisableCopyOnReadDisableCopyOnRead/read_40_disablecopyonread_adam_v_dense_8_kernel"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp/read_40_disablecopyonread_adam_v_dense_8_kernel^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@ *
dtype0o
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@ e
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes

:@ �
Read_41/DisableCopyOnReadDisableCopyOnRead-read_41_disablecopyonread_adam_m_dense_8_bias"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp-read_41_disablecopyonread_adam_m_dense_8_bias^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_42/DisableCopyOnReadDisableCopyOnRead-read_42_disablecopyonread_adam_v_dense_8_bias"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp-read_42_disablecopyonread_adam_v_dense_8_bias^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_43/DisableCopyOnReadDisableCopyOnRead/read_43_disablecopyonread_adam_m_dense_9_kernel"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp/read_43_disablecopyonread_adam_m_dense_9_kernel^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:`*
dtype0o
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:`e
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes

:`�
Read_44/DisableCopyOnReadDisableCopyOnRead/read_44_disablecopyonread_adam_v_dense_9_kernel"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp/read_44_disablecopyonread_adam_v_dense_9_kernel^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:`*
dtype0o
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:`e
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes

:`�
Read_45/DisableCopyOnReadDisableCopyOnRead-read_45_disablecopyonread_adam_m_dense_9_bias"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp-read_45_disablecopyonread_adam_m_dense_9_bias^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_46/DisableCopyOnReadDisableCopyOnRead-read_46_disablecopyonread_adam_v_dense_9_bias"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp-read_46_disablecopyonread_adam_v_dense_9_bias^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_47/DisableCopyOnReadDisableCopyOnRead!read_47_disablecopyonread_total_3"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp!read_47_disablecopyonread_total_3^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_48/DisableCopyOnReadDisableCopyOnRead!read_48_disablecopyonread_count_3"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp!read_48_disablecopyonread_count_3^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_49/DisableCopyOnReadDisableCopyOnRead!read_49_disablecopyonread_total_2"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp!read_49_disablecopyonread_total_2^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_50/DisableCopyOnReadDisableCopyOnRead!read_50_disablecopyonread_count_2"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp!read_50_disablecopyonread_count_2^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_51/DisableCopyOnReadDisableCopyOnRead!read_51_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp!read_51_disablecopyonread_total_1^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_52/DisableCopyOnReadDisableCopyOnRead!read_52_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOp!read_52_disablecopyonread_count_1^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_53/DisableCopyOnReadDisableCopyOnReadread_53_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOpread_53_disablecopyonread_total^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_54/DisableCopyOnReadDisableCopyOnReadread_54_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOpread_54_disablecopyonread_count^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*�
value�B�8B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*�
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *F
dtypes<
:28	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_110Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_111IdentityIdentity_110:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_111Identity_111:output:0*(
_construction_contextkEagerRuntime*�
_input_shapest
r: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:95
3
_user_specified_nameuser_embedding/embeddings:A=
;
_user_specified_name#!product_name_embedding/embeddings:>:
8
_user_specified_name skin_tone_embedding/embeddings:>:
8
_user_specified_name skin_type_embedding/embeddings::6
4
_user_specified_namebrand_embedding/embeddings:.*
(
_user_specified_namedense_6/kernel:,(
&
_user_specified_namedense_6/bias:.*
(
_user_specified_namedense_7/kernel:,	(
&
_user_specified_namedense_7/bias:.
*
(
_user_specified_namedense_5/kernel:,(
&
_user_specified_namedense_5/bias:.*
(
_user_specified_namedense_8/kernel:,(
&
_user_specified_namedense_8/bias:.*
(
_user_specified_namedense_9/kernel:,(
&
_user_specified_namedense_9/bias:)%
#
_user_specified_name	iteration:-)
'
_user_specified_namelearning_rate:@<
:
_user_specified_name" Adam/m/user_embedding/embeddings:@<
:
_user_specified_name" Adam/v/user_embedding/embeddings:HD
B
_user_specified_name*(Adam/m/product_name_embedding/embeddings:HD
B
_user_specified_name*(Adam/v/product_name_embedding/embeddings:EA
?
_user_specified_name'%Adam/m/skin_tone_embedding/embeddings:EA
?
_user_specified_name'%Adam/v/skin_tone_embedding/embeddings:EA
?
_user_specified_name'%Adam/m/skin_type_embedding/embeddings:EA
?
_user_specified_name'%Adam/v/skin_type_embedding/embeddings:A=
;
_user_specified_name#!Adam/m/brand_embedding/embeddings:A=
;
_user_specified_name#!Adam/v/brand_embedding/embeddings:51
/
_user_specified_nameAdam/m/dense_6/kernel:51
/
_user_specified_nameAdam/v/dense_6/kernel:3/
-
_user_specified_nameAdam/m/dense_6/bias:3/
-
_user_specified_nameAdam/v/dense_6/bias:5 1
/
_user_specified_nameAdam/m/dense_7/kernel:5!1
/
_user_specified_nameAdam/v/dense_7/kernel:3"/
-
_user_specified_nameAdam/m/dense_7/bias:3#/
-
_user_specified_nameAdam/v/dense_7/bias:5$1
/
_user_specified_nameAdam/m/dense_5/kernel:5%1
/
_user_specified_nameAdam/v/dense_5/kernel:3&/
-
_user_specified_nameAdam/m/dense_5/bias:3'/
-
_user_specified_nameAdam/v/dense_5/bias:5(1
/
_user_specified_nameAdam/m/dense_8/kernel:5)1
/
_user_specified_nameAdam/v/dense_8/kernel:3*/
-
_user_specified_nameAdam/m/dense_8/bias:3+/
-
_user_specified_nameAdam/v/dense_8/bias:5,1
/
_user_specified_nameAdam/m/dense_9/kernel:5-1
/
_user_specified_nameAdam/v/dense_9/kernel:3./
-
_user_specified_nameAdam/m/dense_9/bias:3//
-
_user_specified_nameAdam/v/dense_9/bias:'0#
!
_user_specified_name	total_3:'1#
!
_user_specified_name	count_3:'2#
!
_user_specified_name	total_2:'3#
!
_user_specified_name	count_2:'4#
!
_user_specified_name	total_1:'5#
!
_user_specified_name	count_1:%6!

_user_specified_nametotal:%7!

_user_specified_namecount:=89

_output_shapes
: 

_user_specified_nameConst
�
F
*__inference_flatten_5_layer_call_fn_452342

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_451789`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2:S O
+
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
I__inference_concatenate_2_layer_call_and_return_conditional_losses_452421
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9concat/axis:output:0*
N
*
T0*(
_output_shapes
:����������X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������2:���������
:���������
:���������2:���������2:���������:���������:���������:����������:����������:Q M
'
_output_shapes
:���������2
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:���������

"
_user_specified_name
inputs_1:QM
'
_output_shapes
:���������

"
_user_specified_name
inputs_2:QM
'
_output_shapes
:���������2
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:���������2
"
_user_specified_name
inputs_4:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_5:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_6:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_7:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs_8:R	N
(
_output_shapes
:����������
"
_user_specified_name
inputs_9
�
�
(__inference_dense_7_layer_call_fn_452462

inputs
unknown:	�@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_451861o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:&"
 
_user_specified_name452456:&"
 
_user_specified_name452458
�

�
C__inference_dense_6_layer_call_and_return_conditional_losses_451845

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
/__inference_user_embedding_layer_call_fn_452264

inputs
unknown:	�C2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_user_embedding_layer_call_and_return_conditional_losses_451780s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������2<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:&"
 
_user_specified_name452260
�
Z
.__inference_concatenate_3_layer_call_fn_452519
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_451912`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������@:��������� :Q M
'
_output_shapes
:���������@
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:��������� 
"
_user_specified_name
inputs_1
�
a
E__inference_flatten_7_layer_call_and_return_conditional_losses_452381

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����
   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������
X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
:S O
+
_output_shapes
:���������

 
_user_specified_nameinputs
�
a
E__inference_flatten_6_layer_call_and_return_conditional_losses_452370

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����
   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������
X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
:S O
+
_output_shapes
:���������

 
_user_specified_nameinputs
��
�"
"__inference__traced_restore_453080
file_prefix=
*assignvariableop_user_embedding_embeddings:	�C2G
4assignvariableop_1_product_name_embedding_embeddings:	�2C
1assignvariableop_2_skin_tone_embedding_embeddings:
C
1assignvariableop_3_skin_type_embedding_embeddings:
?
-assignvariableop_4_brand_embedding_embeddings:N25
!assignvariableop_5_dense_6_kernel:
��.
assignvariableop_6_dense_6_bias:	�4
!assignvariableop_7_dense_7_kernel:	�@-
assignvariableop_8_dense_7_bias:@3
!assignvariableop_9_dense_5_kernel:2@.
 assignvariableop_10_dense_5_bias:@4
"assignvariableop_11_dense_8_kernel:@ .
 assignvariableop_12_dense_8_bias: 4
"assignvariableop_13_dense_9_kernel:`.
 assignvariableop_14_dense_9_bias:'
assignvariableop_15_iteration:	 +
!assignvariableop_16_learning_rate: G
4assignvariableop_17_adam_m_user_embedding_embeddings:	�C2G
4assignvariableop_18_adam_v_user_embedding_embeddings:	�C2O
<assignvariableop_19_adam_m_product_name_embedding_embeddings:	�2O
<assignvariableop_20_adam_v_product_name_embedding_embeddings:	�2K
9assignvariableop_21_adam_m_skin_tone_embedding_embeddings:
K
9assignvariableop_22_adam_v_skin_tone_embedding_embeddings:
K
9assignvariableop_23_adam_m_skin_type_embedding_embeddings:
K
9assignvariableop_24_adam_v_skin_type_embedding_embeddings:
G
5assignvariableop_25_adam_m_brand_embedding_embeddings:N2G
5assignvariableop_26_adam_v_brand_embedding_embeddings:N2=
)assignvariableop_27_adam_m_dense_6_kernel:
��=
)assignvariableop_28_adam_v_dense_6_kernel:
��6
'assignvariableop_29_adam_m_dense_6_bias:	�6
'assignvariableop_30_adam_v_dense_6_bias:	�<
)assignvariableop_31_adam_m_dense_7_kernel:	�@<
)assignvariableop_32_adam_v_dense_7_kernel:	�@5
'assignvariableop_33_adam_m_dense_7_bias:@5
'assignvariableop_34_adam_v_dense_7_bias:@;
)assignvariableop_35_adam_m_dense_5_kernel:2@;
)assignvariableop_36_adam_v_dense_5_kernel:2@5
'assignvariableop_37_adam_m_dense_5_bias:@5
'assignvariableop_38_adam_v_dense_5_bias:@;
)assignvariableop_39_adam_m_dense_8_kernel:@ ;
)assignvariableop_40_adam_v_dense_8_kernel:@ 5
'assignvariableop_41_adam_m_dense_8_bias: 5
'assignvariableop_42_adam_v_dense_8_bias: ;
)assignvariableop_43_adam_m_dense_9_kernel:`;
)assignvariableop_44_adam_v_dense_9_kernel:`5
'assignvariableop_45_adam_m_dense_9_bias:5
'assignvariableop_46_adam_v_dense_9_bias:%
assignvariableop_47_total_3: %
assignvariableop_48_count_3: %
assignvariableop_49_total_2: %
assignvariableop_50_count_2: %
assignvariableop_51_total_1: %
assignvariableop_52_count_1: #
assignvariableop_53_total: #
assignvariableop_54_count: 
identity_56��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*�
value�B�8B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*�
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::*F
dtypes<
:28	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp*assignvariableop_user_embedding_embeddingsIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp4assignvariableop_1_product_name_embedding_embeddingsIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp1assignvariableop_2_skin_tone_embedding_embeddingsIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp1assignvariableop_3_skin_type_embedding_embeddingsIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp-assignvariableop_4_brand_embedding_embeddingsIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_6_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_6_biasIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_7_kernelIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_7_biasIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp!assignvariableop_9_dense_5_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp assignvariableop_10_dense_5_biasIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp"assignvariableop_11_dense_8_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp assignvariableop_12_dense_8_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp"assignvariableop_13_dense_9_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp assignvariableop_14_dense_9_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_iterationIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp!assignvariableop_16_learning_rateIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp4assignvariableop_17_adam_m_user_embedding_embeddingsIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp4assignvariableop_18_adam_v_user_embedding_embeddingsIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp<assignvariableop_19_adam_m_product_name_embedding_embeddingsIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp<assignvariableop_20_adam_v_product_name_embedding_embeddingsIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp9assignvariableop_21_adam_m_skin_tone_embedding_embeddingsIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp9assignvariableop_22_adam_v_skin_tone_embedding_embeddingsIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp9assignvariableop_23_adam_m_skin_type_embedding_embeddingsIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp9assignvariableop_24_adam_v_skin_type_embedding_embeddingsIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp5assignvariableop_25_adam_m_brand_embedding_embeddingsIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp5assignvariableop_26_adam_v_brand_embedding_embeddingsIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_m_dense_6_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_v_dense_6_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_m_dense_6_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_v_dense_6_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_m_dense_7_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_v_dense_7_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp'assignvariableop_33_adam_m_dense_7_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_v_dense_7_biasIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_m_dense_5_kernelIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_v_dense_5_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp'assignvariableop_37_adam_m_dense_5_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp'assignvariableop_38_adam_v_dense_5_biasIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_m_dense_8_kernelIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_v_dense_8_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp'assignvariableop_41_adam_m_dense_8_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_v_dense_8_biasIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_m_dense_9_kernelIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_v_dense_9_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp'assignvariableop_45_adam_m_dense_9_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp'assignvariableop_46_adam_v_dense_9_biasIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOpassignvariableop_47_total_3Identity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOpassignvariableop_48_count_3Identity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOpassignvariableop_49_total_2Identity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOpassignvariableop_50_count_2Identity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOpassignvariableop_51_total_1Identity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOpassignvariableop_52_count_1Identity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOpassignvariableop_53_totalIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOpassignvariableop_54_countIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �

Identity_55Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_56IdentityIdentity_55:output:0^NoOp_1*
T0*
_output_shapes
: �	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_56Identity_56:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesr
p: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:95
3
_user_specified_nameuser_embedding/embeddings:A=
;
_user_specified_name#!product_name_embedding/embeddings:>:
8
_user_specified_name skin_tone_embedding/embeddings:>:
8
_user_specified_name skin_type_embedding/embeddings::6
4
_user_specified_namebrand_embedding/embeddings:.*
(
_user_specified_namedense_6/kernel:,(
&
_user_specified_namedense_6/bias:.*
(
_user_specified_namedense_7/kernel:,	(
&
_user_specified_namedense_7/bias:.
*
(
_user_specified_namedense_5/kernel:,(
&
_user_specified_namedense_5/bias:.*
(
_user_specified_namedense_8/kernel:,(
&
_user_specified_namedense_8/bias:.*
(
_user_specified_namedense_9/kernel:,(
&
_user_specified_namedense_9/bias:)%
#
_user_specified_name	iteration:-)
'
_user_specified_namelearning_rate:@<
:
_user_specified_name" Adam/m/user_embedding/embeddings:@<
:
_user_specified_name" Adam/v/user_embedding/embeddings:HD
B
_user_specified_name*(Adam/m/product_name_embedding/embeddings:HD
B
_user_specified_name*(Adam/v/product_name_embedding/embeddings:EA
?
_user_specified_name'%Adam/m/skin_tone_embedding/embeddings:EA
?
_user_specified_name'%Adam/v/skin_tone_embedding/embeddings:EA
?
_user_specified_name'%Adam/m/skin_type_embedding/embeddings:EA
?
_user_specified_name'%Adam/v/skin_type_embedding/embeddings:A=
;
_user_specified_name#!Adam/m/brand_embedding/embeddings:A=
;
_user_specified_name#!Adam/v/brand_embedding/embeddings:51
/
_user_specified_nameAdam/m/dense_6/kernel:51
/
_user_specified_nameAdam/v/dense_6/kernel:3/
-
_user_specified_nameAdam/m/dense_6/bias:3/
-
_user_specified_nameAdam/v/dense_6/bias:5 1
/
_user_specified_nameAdam/m/dense_7/kernel:5!1
/
_user_specified_nameAdam/v/dense_7/kernel:3"/
-
_user_specified_nameAdam/m/dense_7/bias:3#/
-
_user_specified_nameAdam/v/dense_7/bias:5$1
/
_user_specified_nameAdam/m/dense_5/kernel:5%1
/
_user_specified_nameAdam/v/dense_5/kernel:3&/
-
_user_specified_nameAdam/m/dense_5/bias:3'/
-
_user_specified_nameAdam/v/dense_5/bias:5(1
/
_user_specified_nameAdam/m/dense_8/kernel:5)1
/
_user_specified_nameAdam/v/dense_8/kernel:3*/
-
_user_specified_nameAdam/m/dense_8/bias:3+/
-
_user_specified_nameAdam/v/dense_8/bias:5,1
/
_user_specified_nameAdam/m/dense_9/kernel:5-1
/
_user_specified_nameAdam/v/dense_9/kernel:3./
-
_user_specified_nameAdam/m/dense_9/bias:3//
-
_user_specified_nameAdam/v/dense_9/bias:'0#
!
_user_specified_name	total_3:'1#
!
_user_specified_name	count_3:'2#
!
_user_specified_name	total_2:'3#
!
_user_specified_name	count_2:'4#
!
_user_specified_name	total_1:'5#
!
_user_specified_name	count_1:%6!

_user_specified_nametotal:%7!

_user_specified_namecount
�
�
J__inference_user_embedding_layer_call_and_return_conditional_losses_451780

inputs*
embedding_lookup_451775:	�C2
identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:����������
embedding_lookupResourceGatherembedding_lookup_451775Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/451775*+
_output_shapes
:���������2*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:���������2u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:���������25
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:&"
 
_user_specified_name451775
�Y
�	
C__inference_model_1_layer_call_and_return_conditional_losses_451991

user_input
skin_tone_input
skin_type_input
brand_input
product_name_input
price_input
rating_input
loves_input
review_input
ingredients_input(
brand_embedding_451942:N2,
skin_type_embedding_451945:
,
skin_tone_embedding_451948:
0
product_name_embedding_451951:	�2(
user_embedding_451954:	�C2"
dense_6_451963:
��
dense_6_451965:	�!
dense_7_451968:	�@
dense_7_451970:@ 
dense_5_451974:2@
dense_5_451976:@ 
dense_8_451979:@ 
dense_8_451981:  
dense_9_451985:`
dense_9_451987:
identity��'brand_embedding/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�.product_name_embedding/StatefulPartitionedCall�+skin_tone_embedding/StatefulPartitionedCall�+skin_type_embedding/StatefulPartitionedCall�&user_embedding/StatefulPartitionedCall�
'brand_embedding/StatefulPartitionedCallStatefulPartitionedCallbrand_inputbrand_embedding_451942*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_brand_embedding_layer_call_and_return_conditional_losses_451732�
+skin_type_embedding/StatefulPartitionedCallStatefulPartitionedCallskin_type_inputskin_type_embedding_451945*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_skin_type_embedding_layer_call_and_return_conditional_losses_451744�
+skin_tone_embedding/StatefulPartitionedCallStatefulPartitionedCallskin_tone_inputskin_tone_embedding_451948*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_skin_tone_embedding_layer_call_and_return_conditional_losses_451756�
.product_name_embedding/StatefulPartitionedCallStatefulPartitionedCallproduct_name_inputproduct_name_embedding_451951*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_product_name_embedding_layer_call_and_return_conditional_losses_451768�
&user_embedding/StatefulPartitionedCallStatefulPartitionedCall
user_inputuser_embedding_451954*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_user_embedding_layer_call_and_return_conditional_losses_451780�
flatten_5/PartitionedCallPartitionedCall/user_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_451789�
flatten_6/PartitionedCallPartitionedCall4skin_tone_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_flatten_6_layer_call_and_return_conditional_losses_451796�
flatten_7/PartitionedCallPartitionedCall4skin_type_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_flatten_7_layer_call_and_return_conditional_losses_451803�
flatten_8/PartitionedCallPartitionedCall0brand_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_flatten_8_layer_call_and_return_conditional_losses_451810�
flatten_9/PartitionedCallPartitionedCall7product_name_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_flatten_9_layer_call_and_return_conditional_losses_451817�
concatenate_2/PartitionedCallPartitionedCall"flatten_5/PartitionedCall:output:0"flatten_6/PartitionedCall:output:0"flatten_7/PartitionedCall:output:0"flatten_8/PartitionedCall:output:0"flatten_9/PartitionedCall:output:0price_inputrating_inputloves_inputreview_inputingredients_input*
Tin
2
*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_concatenate_2_layer_call_and_return_conditional_losses_451833�
dense_6/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_6_451963dense_6_451965*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_451845�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_451968dense_7_451970*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_451861�
multiply_1/PartitionedCallPartitionedCall"flatten_5/PartitionedCall:output:0"flatten_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_multiply_1_layer_call_and_return_conditional_losses_451872�
dense_5/StatefulPartitionedCallStatefulPartitionedCall#multiply_1/PartitionedCall:output:0dense_5_451974dense_5_451976*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_451884�
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_451979dense_8_451981*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_451900�
concatenate_3/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_451912�
dense_9/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0dense_9_451985dense_9_451987*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_451923w
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp(^brand_embedding/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall/^product_name_embedding/StatefulPartitionedCall,^skin_tone_embedding/StatefulPartitionedCall,^skin_type_embedding/StatefulPartitionedCall'^user_embedding/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:����������:����������: : : : : : : : : : : : : : : 2R
'brand_embedding/StatefulPartitionedCall'brand_embedding/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2`
.product_name_embedding/StatefulPartitionedCall.product_name_embedding/StatefulPartitionedCall2Z
+skin_tone_embedding/StatefulPartitionedCall+skin_tone_embedding/StatefulPartitionedCall2Z
+skin_type_embedding/StatefulPartitionedCall+skin_type_embedding/StatefulPartitionedCall2P
&user_embedding/StatefulPartitionedCall&user_embedding/StatefulPartitionedCall:S O
'
_output_shapes
:���������
$
_user_specified_name
user_input:XT
'
_output_shapes
:���������
)
_user_specified_nameskin_tone_input:XT
'
_output_shapes
:���������
)
_user_specified_nameskin_type_input:TP
'
_output_shapes
:���������
%
_user_specified_namebrand_input:[W
'
_output_shapes
:���������
,
_user_specified_nameproduct_name_input:TP
'
_output_shapes
:���������
%
_user_specified_nameprice_input:UQ
'
_output_shapes
:���������
&
_user_specified_namerating_input:TP
'
_output_shapes
:���������
%
_user_specified_nameloves_input:VR
(
_output_shapes
:����������
&
_user_specified_namereview_input:[	W
(
_output_shapes
:����������
+
_user_specified_nameingredients_input:&
"
 
_user_specified_name451942:&"
 
_user_specified_name451945:&"
 
_user_specified_name451948:&"
 
_user_specified_name451951:&"
 
_user_specified_name451954:&"
 
_user_specified_name451963:&"
 
_user_specified_name451965:&"
 
_user_specified_name451968:&"
 
_user_specified_name451970:&"
 
_user_specified_name451974:&"
 
_user_specified_name451976:&"
 
_user_specified_name451979:&"
 
_user_specified_name451981:&"
 
_user_specified_name451985:&"
 
_user_specified_name451987
�
�
I__inference_concatenate_2_layer_call_and_return_conditional_losses_451833

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9concat/axis:output:0*
N
*
T0*(
_output_shapes
:����������X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������2:���������
:���������
:���������2:���������2:���������:���������:���������:����������:����������:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������

 
_user_specified_nameinputs:OK
'
_output_shapes
:���������

 
_user_specified_nameinputs:OK
'
_output_shapes
:���������2
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������2
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs:P	L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
F
*__inference_flatten_6_layer_call_fn_452364

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_flatten_6_layer_call_and_return_conditional_losses_451796`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
:S O
+
_output_shapes
:���������

 
_user_specified_nameinputs
�s
�
!__inference__wrapped_model_451712

user_input
skin_tone_input
skin_type_input
brand_input
product_name_input
price_input
rating_input
loves_input
review_input
ingredients_inputA
/model_1_brand_embedding_embedding_lookup_451638:N2E
3model_1_skin_type_embedding_embedding_lookup_451643:
E
3model_1_skin_tone_embedding_embedding_lookup_451648:
I
6model_1_product_name_embedding_embedding_lookup_451653:	�2A
.model_1_user_embedding_embedding_lookup_451658:	�C2B
.model_1_dense_6_matmul_readvariableop_resource:
��>
/model_1_dense_6_biasadd_readvariableop_resource:	�A
.model_1_dense_7_matmul_readvariableop_resource:	�@=
/model_1_dense_7_biasadd_readvariableop_resource:@@
.model_1_dense_5_matmul_readvariableop_resource:2@=
/model_1_dense_5_biasadd_readvariableop_resource:@@
.model_1_dense_8_matmul_readvariableop_resource:@ =
/model_1_dense_8_biasadd_readvariableop_resource: @
.model_1_dense_9_matmul_readvariableop_resource:`=
/model_1_dense_9_biasadd_readvariableop_resource:
identity��(model_1/brand_embedding/embedding_lookup�&model_1/dense_5/BiasAdd/ReadVariableOp�%model_1/dense_5/MatMul/ReadVariableOp�&model_1/dense_6/BiasAdd/ReadVariableOp�%model_1/dense_6/MatMul/ReadVariableOp�&model_1/dense_7/BiasAdd/ReadVariableOp�%model_1/dense_7/MatMul/ReadVariableOp�&model_1/dense_8/BiasAdd/ReadVariableOp�%model_1/dense_8/MatMul/ReadVariableOp�&model_1/dense_9/BiasAdd/ReadVariableOp�%model_1/dense_9/MatMul/ReadVariableOp�/model_1/product_name_embedding/embedding_lookup�,model_1/skin_tone_embedding/embedding_lookup�,model_1/skin_type_embedding/embedding_lookup�'model_1/user_embedding/embedding_lookupr
model_1/brand_embedding/CastCastbrand_input*

DstT0*

SrcT0*'
_output_shapes
:����������
(model_1/brand_embedding/embedding_lookupResourceGather/model_1_brand_embedding_embedding_lookup_451638 model_1/brand_embedding/Cast:y:0*
Tindices0*B
_class8
64loc:@model_1/brand_embedding/embedding_lookup/451638*+
_output_shapes
:���������2*
dtype0�
1model_1/brand_embedding/embedding_lookup/IdentityIdentity1model_1/brand_embedding/embedding_lookup:output:0*
T0*+
_output_shapes
:���������2z
 model_1/skin_type_embedding/CastCastskin_type_input*

DstT0*

SrcT0*'
_output_shapes
:����������
,model_1/skin_type_embedding/embedding_lookupResourceGather3model_1_skin_type_embedding_embedding_lookup_451643$model_1/skin_type_embedding/Cast:y:0*
Tindices0*F
_class<
:8loc:@model_1/skin_type_embedding/embedding_lookup/451643*+
_output_shapes
:���������
*
dtype0�
5model_1/skin_type_embedding/embedding_lookup/IdentityIdentity5model_1/skin_type_embedding/embedding_lookup:output:0*
T0*+
_output_shapes
:���������
z
 model_1/skin_tone_embedding/CastCastskin_tone_input*

DstT0*

SrcT0*'
_output_shapes
:����������
,model_1/skin_tone_embedding/embedding_lookupResourceGather3model_1_skin_tone_embedding_embedding_lookup_451648$model_1/skin_tone_embedding/Cast:y:0*
Tindices0*F
_class<
:8loc:@model_1/skin_tone_embedding/embedding_lookup/451648*+
_output_shapes
:���������
*
dtype0�
5model_1/skin_tone_embedding/embedding_lookup/IdentityIdentity5model_1/skin_tone_embedding/embedding_lookup:output:0*
T0*+
_output_shapes
:���������
�
#model_1/product_name_embedding/CastCastproduct_name_input*

DstT0*

SrcT0*'
_output_shapes
:����������
/model_1/product_name_embedding/embedding_lookupResourceGather6model_1_product_name_embedding_embedding_lookup_451653'model_1/product_name_embedding/Cast:y:0*
Tindices0*I
_class?
=;loc:@model_1/product_name_embedding/embedding_lookup/451653*+
_output_shapes
:���������2*
dtype0�
8model_1/product_name_embedding/embedding_lookup/IdentityIdentity8model_1/product_name_embedding/embedding_lookup:output:0*
T0*+
_output_shapes
:���������2p
model_1/user_embedding/CastCast
user_input*

DstT0*

SrcT0*'
_output_shapes
:����������
'model_1/user_embedding/embedding_lookupResourceGather.model_1_user_embedding_embedding_lookup_451658model_1/user_embedding/Cast:y:0*
Tindices0*A
_class7
53loc:@model_1/user_embedding/embedding_lookup/451658*+
_output_shapes
:���������2*
dtype0�
0model_1/user_embedding/embedding_lookup/IdentityIdentity0model_1/user_embedding/embedding_lookup:output:0*
T0*+
_output_shapes
:���������2h
model_1/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"����2   �
model_1/flatten_5/ReshapeReshape9model_1/user_embedding/embedding_lookup/Identity:output:0 model_1/flatten_5/Const:output:0*
T0*'
_output_shapes
:���������2h
model_1/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"����
   �
model_1/flatten_6/ReshapeReshape>model_1/skin_tone_embedding/embedding_lookup/Identity:output:0 model_1/flatten_6/Const:output:0*
T0*'
_output_shapes
:���������
h
model_1/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"����
   �
model_1/flatten_7/ReshapeReshape>model_1/skin_type_embedding/embedding_lookup/Identity:output:0 model_1/flatten_7/Const:output:0*
T0*'
_output_shapes
:���������
h
model_1/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"����2   �
model_1/flatten_8/ReshapeReshape:model_1/brand_embedding/embedding_lookup/Identity:output:0 model_1/flatten_8/Const:output:0*
T0*'
_output_shapes
:���������2h
model_1/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"����2   �
model_1/flatten_9/ReshapeReshapeAmodel_1/product_name_embedding/embedding_lookup/Identity:output:0 model_1/flatten_9/Const:output:0*
T0*'
_output_shapes
:���������2c
!model_1/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model_1/concatenate_2/concatConcatV2"model_1/flatten_5/Reshape:output:0"model_1/flatten_6/Reshape:output:0"model_1/flatten_7/Reshape:output:0"model_1/flatten_8/Reshape:output:0"model_1/flatten_9/Reshape:output:0price_inputrating_inputloves_inputreview_inputingredients_input*model_1/concatenate_2/concat/axis:output:0*
N
*
T0*(
_output_shapes
:�����������
%model_1/dense_6/MatMul/ReadVariableOpReadVariableOp.model_1_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model_1/dense_6/MatMulMatMul%model_1/concatenate_2/concat:output:0-model_1/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
&model_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_1/dense_6/BiasAddBiasAdd model_1/dense_6/MatMul:product:0.model_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������q
model_1/dense_6/ReluRelu model_1/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
%model_1/dense_7/MatMul/ReadVariableOpReadVariableOp.model_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
model_1/dense_7/MatMulMatMul"model_1/dense_6/Relu:activations:0-model_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
&model_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model_1/dense_7/BiasAddBiasAdd model_1/dense_7/MatMul:product:0.model_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@p
model_1/dense_7/ReluRelu model_1/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
model_1/multiply_1/mulMul"model_1/flatten_5/Reshape:output:0"model_1/flatten_9/Reshape:output:0*
T0*'
_output_shapes
:���������2�
%model_1/dense_5/MatMul/ReadVariableOpReadVariableOp.model_1_dense_5_matmul_readvariableop_resource*
_output_shapes

:2@*
dtype0�
model_1/dense_5/MatMulMatMulmodel_1/multiply_1/mul:z:0-model_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
&model_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model_1/dense_5/BiasAddBiasAdd model_1/dense_5/MatMul:product:0.model_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@p
model_1/dense_5/ReluRelu model_1/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
%model_1/dense_8/MatMul/ReadVariableOpReadVariableOp.model_1_dense_8_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
model_1/dense_8/MatMulMatMul"model_1/dense_7/Relu:activations:0-model_1/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
&model_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model_1/dense_8/BiasAddBiasAdd model_1/dense_8/MatMul:product:0.model_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� p
model_1/dense_8/ReluRelu model_1/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:��������� c
!model_1/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model_1/concatenate_3/concatConcatV2"model_1/dense_5/Relu:activations:0"model_1/dense_8/Relu:activations:0*model_1/concatenate_3/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������`�
%model_1/dense_9/MatMul/ReadVariableOpReadVariableOp.model_1_dense_9_matmul_readvariableop_resource*
_output_shapes

:`*
dtype0�
model_1/dense_9/MatMulMatMul%model_1/concatenate_3/concat:output:0-model_1/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
&model_1/dense_9/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_1/dense_9/BiasAddBiasAdd model_1/dense_9/MatMul:product:0.model_1/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������o
IdentityIdentity model_1/dense_9/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp)^model_1/brand_embedding/embedding_lookup'^model_1/dense_5/BiasAdd/ReadVariableOp&^model_1/dense_5/MatMul/ReadVariableOp'^model_1/dense_6/BiasAdd/ReadVariableOp&^model_1/dense_6/MatMul/ReadVariableOp'^model_1/dense_7/BiasAdd/ReadVariableOp&^model_1/dense_7/MatMul/ReadVariableOp'^model_1/dense_8/BiasAdd/ReadVariableOp&^model_1/dense_8/MatMul/ReadVariableOp'^model_1/dense_9/BiasAdd/ReadVariableOp&^model_1/dense_9/MatMul/ReadVariableOp0^model_1/product_name_embedding/embedding_lookup-^model_1/skin_tone_embedding/embedding_lookup-^model_1/skin_type_embedding/embedding_lookup(^model_1/user_embedding/embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:����������:����������: : : : : : : : : : : : : : : 2T
(model_1/brand_embedding/embedding_lookup(model_1/brand_embedding/embedding_lookup2P
&model_1/dense_5/BiasAdd/ReadVariableOp&model_1/dense_5/BiasAdd/ReadVariableOp2N
%model_1/dense_5/MatMul/ReadVariableOp%model_1/dense_5/MatMul/ReadVariableOp2P
&model_1/dense_6/BiasAdd/ReadVariableOp&model_1/dense_6/BiasAdd/ReadVariableOp2N
%model_1/dense_6/MatMul/ReadVariableOp%model_1/dense_6/MatMul/ReadVariableOp2P
&model_1/dense_7/BiasAdd/ReadVariableOp&model_1/dense_7/BiasAdd/ReadVariableOp2N
%model_1/dense_7/MatMul/ReadVariableOp%model_1/dense_7/MatMul/ReadVariableOp2P
&model_1/dense_8/BiasAdd/ReadVariableOp&model_1/dense_8/BiasAdd/ReadVariableOp2N
%model_1/dense_8/MatMul/ReadVariableOp%model_1/dense_8/MatMul/ReadVariableOp2P
&model_1/dense_9/BiasAdd/ReadVariableOp&model_1/dense_9/BiasAdd/ReadVariableOp2N
%model_1/dense_9/MatMul/ReadVariableOp%model_1/dense_9/MatMul/ReadVariableOp2b
/model_1/product_name_embedding/embedding_lookup/model_1/product_name_embedding/embedding_lookup2\
,model_1/skin_tone_embedding/embedding_lookup,model_1/skin_tone_embedding/embedding_lookup2\
,model_1/skin_type_embedding/embedding_lookup,model_1/skin_type_embedding/embedding_lookup2R
'model_1/user_embedding/embedding_lookup'model_1/user_embedding/embedding_lookup:S O
'
_output_shapes
:���������
$
_user_specified_name
user_input:XT
'
_output_shapes
:���������
)
_user_specified_nameskin_tone_input:XT
'
_output_shapes
:���������
)
_user_specified_nameskin_type_input:TP
'
_output_shapes
:���������
%
_user_specified_namebrand_input:[W
'
_output_shapes
:���������
,
_user_specified_nameproduct_name_input:TP
'
_output_shapes
:���������
%
_user_specified_nameprice_input:UQ
'
_output_shapes
:���������
&
_user_specified_namerating_input:TP
'
_output_shapes
:���������
%
_user_specified_nameloves_input:VR
(
_output_shapes
:����������
&
_user_specified_namereview_input:[	W
(
_output_shapes
:����������
+
_user_specified_nameingredients_input:&
"
 
_user_specified_name451638:&"
 
_user_specified_name451643:&"
 
_user_specified_name451648:&"
 
_user_specified_name451653:&"
 
_user_specified_name451658:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
a
E__inference_flatten_9_layer_call_and_return_conditional_losses_451817

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����2   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������2X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2:S O
+
_output_shapes
:���������2
 
_user_specified_nameinputs
�
s
I__inference_concatenate_3_layer_call_and_return_conditional_losses_451912

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :u
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:���������`W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������@:��������� :O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
u
I__inference_concatenate_3_layer_call_and_return_conditional_losses_452526
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :w
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:���������`W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������@:��������� :Q M
'
_output_shapes
:���������@
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:��������� 
"
_user_specified_name
inputs_1
�

�
C__inference_dense_5_layer_call_and_return_conditional_losses_451884

inputs0
matmul_readvariableop_resource:2@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
(__inference_dense_8_layer_call_fn_452502

inputs
unknown:@ 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_451900o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:&"
 
_user_specified_name452496:&"
 
_user_specified_name452498
�
F
*__inference_flatten_7_layer_call_fn_452375

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_flatten_7_layer_call_and_return_conditional_losses_451803`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
:S O
+
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
K__inference_brand_embedding_layer_call_and_return_conditional_losses_451732

inputs)
embedding_lookup_451727:N2
identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:����������
embedding_lookupResourceGatherembedding_lookup_451727Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/451727*+
_output_shapes
:���������2*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:���������2u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:���������25
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:&"
 
_user_specified_name451727
�
a
E__inference_flatten_8_layer_call_and_return_conditional_losses_452392

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����2   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������2X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2:S O
+
_output_shapes
:���������2
 
_user_specified_nameinputs
�
a
E__inference_flatten_8_layer_call_and_return_conditional_losses_451810

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����2   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������2X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2:S O
+
_output_shapes
:���������2
 
_user_specified_nameinputs
�
a
E__inference_flatten_9_layer_call_and_return_conditional_losses_452359

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����2   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������2X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2:S O
+
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
.__inference_concatenate_2_layer_call_fn_452406
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9*
Tin
2
*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_concatenate_2_layer_call_and_return_conditional_losses_451833a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������2:���������
:���������
:���������2:���������2:���������:���������:���������:����������:����������:Q M
'
_output_shapes
:���������2
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:���������

"
_user_specified_name
inputs_1:QM
'
_output_shapes
:���������

"
_user_specified_name
inputs_2:QM
'
_output_shapes
:���������2
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:���������2
"
_user_specified_name
inputs_4:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_5:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_6:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_7:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs_8:R	N
(
_output_shapes
:����������
"
_user_specified_name
inputs_9
�

�
C__inference_dense_7_layer_call_and_return_conditional_losses_451861

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
a
E__inference_flatten_6_layer_call_and_return_conditional_losses_451796

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����
   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������
X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
:S O
+
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
R__inference_product_name_embedding_layer_call_and_return_conditional_losses_451768

inputs*
embedding_lookup_451763:	�2
identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:����������
embedding_lookupResourceGatherembedding_lookup_451763Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/451763*+
_output_shapes
:���������2*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:���������2u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:���������25
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:&"
 
_user_specified_name451763
�	
�
C__inference_dense_9_layer_call_and_return_conditional_losses_451923

inputs0
matmul_readvariableop_resource:`-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
O__inference_skin_type_embedding_layer_call_and_return_conditional_losses_452321

inputs)
embedding_lookup_452316:

identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:����������
embedding_lookupResourceGatherembedding_lookup_452316Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/452316*+
_output_shapes
:���������
*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:���������
u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:���������
5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:&"
 
_user_specified_name452316
�
r
F__inference_multiply_1_layer_call_and_return_conditional_losses_452453
inputs_0
inputs_1
identityP
mulMulinputs_0inputs_1*
T0*'
_output_shapes
:���������2O
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:���������2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������2:���������2:Q M
'
_output_shapes
:���������2
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:���������2
"
_user_specified_name
inputs_1
�
�
7__inference_product_name_embedding_layer_call_fn_452280

inputs
unknown:	�2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_product_name_embedding_layer_call_and_return_conditional_losses_451768s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������2<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:&"
 
_user_specified_name452276"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
brand_input4
serving_default_brand_input:0���������
P
ingredients_input;
#serving_default_ingredients_input:0����������
C
loves_input4
serving_default_loves_input:0���������
C
price_input4
serving_default_price_input:0���������
Q
product_name_input;
$serving_default_product_name_input:0���������
E
rating_input5
serving_default_rating_input:0���������
F
review_input6
serving_default_review_input:0����������
K
skin_tone_input8
!serving_default_skin_tone_input:0���������
K
skin_type_input8
!serving_default_skin_type_input:0���������
A

user_input3
serving_default_user_input:0���������;
dense_90
StatefulPartitionedCall:0���������tensorflow/serving/predict:Ԏ
�
layer-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-0
layer-5
layer_with_weights-1
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer_with_weights-5
layer-21
layer-22
layer_with_weights-6
layer-23
layer_with_weights-7
layer-24
layer_with_weights-8
layer-25
layer-26
layer_with_weights-9
layer-27
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
#_default_save_signature
$	optimizer
%
signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
�
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
,
embeddings"
_tf_keras_layer
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses
3
embeddings"
_tf_keras_layer
�
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses
:
embeddings"
_tf_keras_layer
�
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses
A
embeddings"
_tf_keras_layer
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
H
embeddings"
_tf_keras_layer
�
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses"
_tf_keras_layer
�
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses"
_tf_keras_layer
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"
_tf_keras_layer
�
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses"
_tf_keras_layer
�
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
�
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses"
_tf_keras_layer
�
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses

skernel
tbias"
_tf_keras_layer
�
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses"
_tf_keras_layer
�
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
,0
31
:2
A3
H4
s5
t6
�7
�8
�9
�10
�11
�12
�13
�14"
trackable_list_wrapper
�
,0
31
:2
A3
H4
s5
t6
�7
�8
�9
�10
�11
�12
�13
�14"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
#_default_save_signature
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
(__inference_model_1_layer_call_fn_452035
(__inference_model_1_layer_call_fn_452079�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
C__inference_model_1_layer_call_and_return_conditional_losses_451930
C__inference_model_1_layer_call_and_return_conditional_losses_451991�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�B�
!__inference__wrapped_model_451712
user_inputskin_tone_inputskin_type_inputbrand_inputproduct_name_inputprice_inputrating_inputloves_inputreview_inputingredients_input
"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla"
experimentalOptimizer
-
�serving_default"
signature_map
'
,0"
trackable_list_wrapper
'
,0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_user_embedding_layer_call_fn_452264�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
J__inference_user_embedding_layer_call_and_return_conditional_losses_452273�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
,:*	�C22user_embedding/embeddings
'
30"
trackable_list_wrapper
'
30"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
7__inference_product_name_embedding_layer_call_fn_452280�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
R__inference_product_name_embedding_layer_call_and_return_conditional_losses_452289�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
4:2	�22!product_name_embedding/embeddings
'
:0"
trackable_list_wrapper
'
:0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
4__inference_skin_tone_embedding_layer_call_fn_452296�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
O__inference_skin_tone_embedding_layer_call_and_return_conditional_losses_452305�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0:.
2skin_tone_embedding/embeddings
'
A0"
trackable_list_wrapper
'
A0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
4__inference_skin_type_embedding_layer_call_fn_452312�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
O__inference_skin_type_embedding_layer_call_and_return_conditional_losses_452321�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0:.
2skin_type_embedding/embeddings
'
H0"
trackable_list_wrapper
'
H0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_brand_embedding_layer_call_fn_452328�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_brand_embedding_layer_call_and_return_conditional_losses_452337�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
,:*N22brand_embedding/embeddings
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_flatten_5_layer_call_fn_452342�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_flatten_5_layer_call_and_return_conditional_losses_452348�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_flatten_9_layer_call_fn_452353�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_flatten_9_layer_call_and_return_conditional_losses_452359�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_flatten_6_layer_call_fn_452364�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_flatten_6_layer_call_and_return_conditional_losses_452370�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_flatten_7_layer_call_fn_452375�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_flatten_7_layer_call_and_return_conditional_losses_452381�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_flatten_8_layer_call_fn_452386�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_flatten_8_layer_call_and_return_conditional_losses_452392�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_concatenate_2_layer_call_fn_452406�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_concatenate_2_layer_call_and_return_conditional_losses_452421�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
s0
t1"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_6_layer_call_fn_452430�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_6_layer_call_and_return_conditional_losses_452441�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
": 
��2dense_6/kernel
:�2dense_6/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_multiply_1_layer_call_fn_452447�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_multiply_1_layer_call_and_return_conditional_losses_452453�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_7_layer_call_fn_452462�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_7_layer_call_and_return_conditional_losses_452473�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:	�@2dense_7/kernel
:@2dense_7/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_5_layer_call_fn_452482�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_5_layer_call_and_return_conditional_losses_452493�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 :2@2dense_5/kernel
:@2dense_5/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_8_layer_call_fn_452502�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_8_layer_call_and_return_conditional_losses_452513�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 :@ 2dense_8/kernel
: 2dense_8/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_concatenate_3_layer_call_fn_452519�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_concatenate_3_layer_call_and_return_conditional_losses_452526�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_9_layer_call_fn_452535�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_9_layer_call_and_return_conditional_losses_452545�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 :`2dense_9/kernel
:2dense_9/bias
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27"
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_model_1_layer_call_fn_452035
user_inputskin_tone_inputskin_type_inputbrand_inputproduct_name_inputprice_inputrating_inputloves_inputreview_inputingredients_input
"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_model_1_layer_call_fn_452079
user_inputskin_tone_inputskin_type_inputbrand_inputproduct_name_inputprice_inputrating_inputloves_inputreview_inputingredients_input
"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_model_1_layer_call_and_return_conditional_losses_451930
user_inputskin_tone_inputskin_type_inputbrand_inputproduct_name_inputprice_inputrating_inputloves_inputreview_inputingredients_input
"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_model_1_layer_call_and_return_conditional_losses_451991
user_inputskin_tone_inputskin_type_inputbrand_inputproduct_name_inputprice_inputrating_inputloves_inputreview_inputingredients_input
"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
$__inference_signature_wrapper_452257brand_inputingredients_inputloves_inputprice_inputproduct_name_inputrating_inputreview_inputskin_tone_inputskin_type_input
user_input"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 �

kwonlyargs���
jbrand_input
jingredients_input
jloves_input
jprice_input
jproduct_name_input
jrating_input
jreview_input
jskin_tone_input
jskin_type_input
j
user_input
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
/__inference_user_embedding_layer_call_fn_452264inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_user_embedding_layer_call_and_return_conditional_losses_452273inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
7__inference_product_name_embedding_layer_call_fn_452280inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_product_name_embedding_layer_call_and_return_conditional_losses_452289inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
4__inference_skin_tone_embedding_layer_call_fn_452296inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_skin_tone_embedding_layer_call_and_return_conditional_losses_452305inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
4__inference_skin_type_embedding_layer_call_fn_452312inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_skin_type_embedding_layer_call_and_return_conditional_losses_452321inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_brand_embedding_layer_call_fn_452328inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_brand_embedding_layer_call_and_return_conditional_losses_452337inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_flatten_5_layer_call_fn_452342inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_flatten_5_layer_call_and_return_conditional_losses_452348inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_flatten_9_layer_call_fn_452353inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_flatten_9_layer_call_and_return_conditional_losses_452359inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_flatten_6_layer_call_fn_452364inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_flatten_6_layer_call_and_return_conditional_losses_452370inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_flatten_7_layer_call_fn_452375inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_flatten_7_layer_call_and_return_conditional_losses_452381inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_flatten_8_layer_call_fn_452386inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_flatten_8_layer_call_and_return_conditional_losses_452392inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_concatenate_2_layer_call_fn_452406inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9
"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_concatenate_2_layer_call_and_return_conditional_losses_452421inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9
"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_6_layer_call_fn_452430inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_6_layer_call_and_return_conditional_losses_452441inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_multiply_1_layer_call_fn_452447inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_multiply_1_layer_call_and_return_conditional_losses_452453inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_7_layer_call_fn_452462inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_7_layer_call_and_return_conditional_losses_452473inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_5_layer_call_fn_452482inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_5_layer_call_and_return_conditional_losses_452493inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_8_layer_call_fn_452502inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_8_layer_call_and_return_conditional_losses_452513inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_concatenate_3_layer_call_fn_452519inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_concatenate_3_layer_call_and_return_conditional_losses_452526inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_9_layer_call_fn_452535inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_9_layer_call_and_return_conditional_losses_452545inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
1:/	�C22 Adam/m/user_embedding/embeddings
1:/	�C22 Adam/v/user_embedding/embeddings
9:7	�22(Adam/m/product_name_embedding/embeddings
9:7	�22(Adam/v/product_name_embedding/embeddings
5:3
2%Adam/m/skin_tone_embedding/embeddings
5:3
2%Adam/v/skin_tone_embedding/embeddings
5:3
2%Adam/m/skin_type_embedding/embeddings
5:3
2%Adam/v/skin_type_embedding/embeddings
1:/N22!Adam/m/brand_embedding/embeddings
1:/N22!Adam/v/brand_embedding/embeddings
':%
��2Adam/m/dense_6/kernel
':%
��2Adam/v/dense_6/kernel
 :�2Adam/m/dense_6/bias
 :�2Adam/v/dense_6/bias
&:$	�@2Adam/m/dense_7/kernel
&:$	�@2Adam/v/dense_7/kernel
:@2Adam/m/dense_7/bias
:@2Adam/v/dense_7/bias
%:#2@2Adam/m/dense_5/kernel
%:#2@2Adam/v/dense_5/kernel
:@2Adam/m/dense_5/bias
:@2Adam/v/dense_5/bias
%:#@ 2Adam/m/dense_8/kernel
%:#@ 2Adam/v/dense_8/kernel
: 2Adam/m/dense_8/bias
: 2Adam/v/dense_8/bias
%:#`2Adam/m/dense_9/kernel
%:#`2Adam/v/dense_9/kernel
:2Adam/m/dense_9/bias
:2Adam/v/dense_9/bias
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
!__inference__wrapped_model_451712�HA:3,st�����������
���
���
$�!

user_input���������
)�&
skin_tone_input���������
)�&
skin_type_input���������
%�"
brand_input���������
,�)
product_name_input���������
%�"
price_input���������
&�#
rating_input���������
%�"
loves_input���������
'�$
review_input����������
,�)
ingredients_input����������
� "1�.
,
dense_9!�
dense_9����������
K__inference_brand_embedding_layer_call_and_return_conditional_losses_452337fH/�,
%�"
 �
inputs���������
� "0�-
&�#
tensor_0���������2
� �
0__inference_brand_embedding_layer_call_fn_452328[H/�,
%�"
 �
inputs���������
� "%�"
unknown���������2�
I__inference_concatenate_2_layer_call_and_return_conditional_losses_452421����
���
���
"�
inputs_0���������2
"�
inputs_1���������

"�
inputs_2���������

"�
inputs_3���������2
"�
inputs_4���������2
"�
inputs_5���������
"�
inputs_6���������
"�
inputs_7���������
#� 
inputs_8����������
#� 
inputs_9����������
� "-�*
#� 
tensor_0����������
� �
.__inference_concatenate_2_layer_call_fn_452406����
���
���
"�
inputs_0���������2
"�
inputs_1���������

"�
inputs_2���������

"�
inputs_3���������2
"�
inputs_4���������2
"�
inputs_5���������
"�
inputs_6���������
"�
inputs_7���������
#� 
inputs_8����������
#� 
inputs_9����������
� ""�
unknown�����������
I__inference_concatenate_3_layer_call_and_return_conditional_losses_452526�Z�W
P�M
K�H
"�
inputs_0���������@
"�
inputs_1��������� 
� ",�)
"�
tensor_0���������`
� �
.__inference_concatenate_3_layer_call_fn_452519Z�W
P�M
K�H
"�
inputs_0���������@
"�
inputs_1��������� 
� "!�
unknown���������`�
C__inference_dense_5_layer_call_and_return_conditional_losses_452493e��/�,
%�"
 �
inputs���������2
� ",�)
"�
tensor_0���������@
� �
(__inference_dense_5_layer_call_fn_452482Z��/�,
%�"
 �
inputs���������2
� "!�
unknown���������@�
C__inference_dense_6_layer_call_and_return_conditional_losses_452441est0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
(__inference_dense_6_layer_call_fn_452430Zst0�-
&�#
!�
inputs����������
� ""�
unknown�����������
C__inference_dense_7_layer_call_and_return_conditional_losses_452473f��0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������@
� �
(__inference_dense_7_layer_call_fn_452462[��0�-
&�#
!�
inputs����������
� "!�
unknown���������@�
C__inference_dense_8_layer_call_and_return_conditional_losses_452513e��/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0��������� 
� �
(__inference_dense_8_layer_call_fn_452502Z��/�,
%�"
 �
inputs���������@
� "!�
unknown��������� �
C__inference_dense_9_layer_call_and_return_conditional_losses_452545e��/�,
%�"
 �
inputs���������`
� ",�)
"�
tensor_0���������
� �
(__inference_dense_9_layer_call_fn_452535Z��/�,
%�"
 �
inputs���������`
� "!�
unknown����������
E__inference_flatten_5_layer_call_and_return_conditional_losses_452348c3�0
)�&
$�!
inputs���������2
� ",�)
"�
tensor_0���������2
� �
*__inference_flatten_5_layer_call_fn_452342X3�0
)�&
$�!
inputs���������2
� "!�
unknown���������2�
E__inference_flatten_6_layer_call_and_return_conditional_losses_452370c3�0
)�&
$�!
inputs���������

� ",�)
"�
tensor_0���������

� �
*__inference_flatten_6_layer_call_fn_452364X3�0
)�&
$�!
inputs���������

� "!�
unknown���������
�
E__inference_flatten_7_layer_call_and_return_conditional_losses_452381c3�0
)�&
$�!
inputs���������

� ",�)
"�
tensor_0���������

� �
*__inference_flatten_7_layer_call_fn_452375X3�0
)�&
$�!
inputs���������

� "!�
unknown���������
�
E__inference_flatten_8_layer_call_and_return_conditional_losses_452392c3�0
)�&
$�!
inputs���������2
� ",�)
"�
tensor_0���������2
� �
*__inference_flatten_8_layer_call_fn_452386X3�0
)�&
$�!
inputs���������2
� "!�
unknown���������2�
E__inference_flatten_9_layer_call_and_return_conditional_losses_452359c3�0
)�&
$�!
inputs���������2
� ",�)
"�
tensor_0���������2
� �
*__inference_flatten_9_layer_call_fn_452353X3�0
)�&
$�!
inputs���������2
� "!�
unknown���������2�
C__inference_model_1_layer_call_and_return_conditional_losses_451930�HA:3,st�����������
���
���
$�!

user_input���������
)�&
skin_tone_input���������
)�&
skin_type_input���������
%�"
brand_input���������
,�)
product_name_input���������
%�"
price_input���������
&�#
rating_input���������
%�"
loves_input���������
'�$
review_input����������
,�)
ingredients_input����������
p

 
� ",�)
"�
tensor_0���������
� �
C__inference_model_1_layer_call_and_return_conditional_losses_451991�HA:3,st�����������
���
���
$�!

user_input���������
)�&
skin_tone_input���������
)�&
skin_type_input���������
%�"
brand_input���������
,�)
product_name_input���������
%�"
price_input���������
&�#
rating_input���������
%�"
loves_input���������
'�$
review_input����������
,�)
ingredients_input����������
p 

 
� ",�)
"�
tensor_0���������
� �
(__inference_model_1_layer_call_fn_452035�HA:3,st�����������
���
���
$�!

user_input���������
)�&
skin_tone_input���������
)�&
skin_type_input���������
%�"
brand_input���������
,�)
product_name_input���������
%�"
price_input���������
&�#
rating_input���������
%�"
loves_input���������
'�$
review_input����������
,�)
ingredients_input����������
p

 
� "!�
unknown����������
(__inference_model_1_layer_call_fn_452079�HA:3,st�����������
���
���
$�!

user_input���������
)�&
skin_tone_input���������
)�&
skin_type_input���������
%�"
brand_input���������
,�)
product_name_input���������
%�"
price_input���������
&�#
rating_input���������
%�"
loves_input���������
'�$
review_input����������
,�)
ingredients_input����������
p 

 
� "!�
unknown����������
F__inference_multiply_1_layer_call_and_return_conditional_losses_452453�Z�W
P�M
K�H
"�
inputs_0���������2
"�
inputs_1���������2
� ",�)
"�
tensor_0���������2
� �
+__inference_multiply_1_layer_call_fn_452447Z�W
P�M
K�H
"�
inputs_0���������2
"�
inputs_1���������2
� "!�
unknown���������2�
R__inference_product_name_embedding_layer_call_and_return_conditional_losses_452289f3/�,
%�"
 �
inputs���������
� "0�-
&�#
tensor_0���������2
� �
7__inference_product_name_embedding_layer_call_fn_452280[3/�,
%�"
 �
inputs���������
� "%�"
unknown���������2�
$__inference_signature_wrapper_452257�HA:3,st�����������
� 
���
4
brand_input%�"
brand_input���������
A
ingredients_input,�)
ingredients_input����������
4
loves_input%�"
loves_input���������
4
price_input%�"
price_input���������
B
product_name_input,�)
product_name_input���������
6
rating_input&�#
rating_input���������
7
review_input'�$
review_input����������
<
skin_tone_input)�&
skin_tone_input���������
<
skin_type_input)�&
skin_type_input���������
2

user_input$�!

user_input���������"1�.
,
dense_9!�
dense_9����������
O__inference_skin_tone_embedding_layer_call_and_return_conditional_losses_452305f:/�,
%�"
 �
inputs���������
� "0�-
&�#
tensor_0���������

� �
4__inference_skin_tone_embedding_layer_call_fn_452296[:/�,
%�"
 �
inputs���������
� "%�"
unknown���������
�
O__inference_skin_type_embedding_layer_call_and_return_conditional_losses_452321fA/�,
%�"
 �
inputs���������
� "0�-
&�#
tensor_0���������

� �
4__inference_skin_type_embedding_layer_call_fn_452312[A/�,
%�"
 �
inputs���������
� "%�"
unknown���������
�
J__inference_user_embedding_layer_call_and_return_conditional_losses_452273f,/�,
%�"
 �
inputs���������
� "0�-
&�#
tensor_0���������2
� �
/__inference_user_embedding_layer_call_fn_452264[,/�,
%�"
 �
inputs���������
� "%�"
unknown���������2