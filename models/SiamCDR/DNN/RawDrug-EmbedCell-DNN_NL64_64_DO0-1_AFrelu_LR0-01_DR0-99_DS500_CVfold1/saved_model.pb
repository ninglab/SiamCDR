��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
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
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
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
0
Sigmoid
x"T
y"T"
Ttype:

2
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
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68�
{
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@* 
shared_namedense_12/kernel
t
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes
:	�@*
dtype0
r
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_12/bias
k
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes
:@*
dtype0
z
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_13/kernel
s
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes

:@@*
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
:@*
dtype0
z
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_14/kernel
s
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*
_output_shapes

:@*
dtype0
r
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_14/bias
k
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	�@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:@*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:@@*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:@*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:@@*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:@*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
�
Adam/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*'
shared_nameAdam/dense_12/kernel/m
�
*Adam/dense_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/m*
_output_shapes
:	�@*
dtype0
�
Adam/dense_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_12/bias/m
y
(Adam/dense_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/m*
_output_shapes
:@*
dtype0
�
Adam/dense_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_13/kernel/m
�
*Adam/dense_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/m*
_output_shapes

:@@*
dtype0
�
Adam/dense_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_13/bias/m
y
(Adam/dense_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/m*
_output_shapes
:@*
dtype0
�
Adam/dense_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_14/kernel/m
�
*Adam/dense_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/m*
_output_shapes

:@*
dtype0
�
Adam/dense_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_14/bias/m
y
(Adam/dense_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*'
shared_nameAdam/dense_12/kernel/v
�
*Adam/dense_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/v*
_output_shapes
:	�@*
dtype0
�
Adam/dense_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_12/bias/v
y
(Adam/dense_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/v*
_output_shapes
:@*
dtype0
�
Adam/dense_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_13/kernel/v
�
*Adam/dense_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/v*
_output_shapes

:@@*
dtype0
�
Adam/dense_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_13/bias/v
y
(Adam/dense_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/v*
_output_shapes
:@*
dtype0
�
Adam/dense_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_14/kernel/v
�
*Adam/dense_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/v*
_output_shapes

:@*
dtype0
�
Adam/dense_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_14/bias/v
y
(Adam/dense_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�\
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�\
value�[B�[ B�[
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
* 
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses*
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&_random_generator
'__call__
*(&call_and_return_all_conditional_losses* 
�

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses*
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5_random_generator
6__call__
*7&call_and_return_all_conditional_losses* 
�

8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses*
�
@iter

Abeta_1

Bbeta_2
	Cdecaym�m�)m�*m�8m�9m�v�v�)v�*v�8v�9v�*
Z
D0
E1
F2
G3
H4
I5
6
7
)8
*9
810
911*
.
0
1
)2
*3
84
95*
* 
�
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Oserving_default* 
�
Player-0
Qlayer_with_weights-0
Qlayer-1
Rlayer-2
Slayer_with_weights-1
Slayer-3
Tlayer-4
Ulayer_with_weights-2
Ulayer-5
#V_self_saveable_object_factories
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses*
�
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses* 
.
D0
E1
F2
G3
H4
I5*
* 
* 
�
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEdense_12/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_12/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
�
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
"	variables
#trainable_variables
$regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses* 
* 
* 
* 
_Y
VARIABLE_VALUEdense_13/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_13/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

)0
*1*

)0
*1*
* 
�
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
1	variables
2trainable_variables
3regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses* 
* 
* 
* 
_Y
VARIABLE_VALUEdense_14/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_14/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

80
91*

80
91*
* 
�
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
�layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
.
D0
E1
F2
G3
H4
I5*
<
0
1
2
3
4
5
6
7*

�0*
* 
* 
* 
(
$�_self_saveable_object_factories* 
�

Dkernel
Ebias
$�_self_saveable_object_factories
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
$�_self_saveable_object_factories
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�_random_generator
�__call__
+�&call_and_return_all_conditional_losses* 
�

Fkernel
Gbias
$�_self_saveable_object_factories
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
$�_self_saveable_object_factories
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�_random_generator
�__call__
+�&call_and_return_all_conditional_losses* 
�

Hkernel
Ibias
$�_self_saveable_object_factories
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
* 
.
D0
E1
F2
G3
H4
I5*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses* 
* 
* 
.
D0
E1
F2
G3
H4
I5*
 
0
1
2
3*
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

�total

�count
�	variables
�	keras_api*
* 
* 

D0
E1*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 

F0
G1*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 

H0
I1*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
.
D0
E1
F2
G3
H4
I5*
.
P0
Q1
R2
S3
T4
U5*
* 
* 
* 
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*

D0
E1*
* 
* 
* 
* 
* 
* 
* 
* 
* 

F0
G1*
* 
* 
* 
* 
* 
* 
* 
* 
* 

H0
I1*
* 
* 
* 
* 
�|
VARIABLE_VALUEAdam/dense_12/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_12/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_13/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_13/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_14/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_14/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_12/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_12/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_13/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_13/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_14/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_14/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
serving_default_input_10Placeholder*(
_output_shapes
:����������*
dtype0*
shape:����������
|
serving_default_input_9Placeholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_10serving_default_input_9dense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_3439311
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOp#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_12/kernel/m/Read/ReadVariableOp(Adam/dense_12/bias/m/Read/ReadVariableOp*Adam/dense_13/kernel/m/Read/ReadVariableOp(Adam/dense_13/bias/m/Read/ReadVariableOp*Adam/dense_14/kernel/m/Read/ReadVariableOp(Adam/dense_14/bias/m/Read/ReadVariableOp*Adam/dense_12/kernel/v/Read/ReadVariableOp(Adam/dense_12/bias/v/Read/ReadVariableOp*Adam/dense_13/kernel/v/Read/ReadVariableOp(Adam/dense_13/bias/v/Read/ReadVariableOp*Adam/dense_14/kernel/v/Read/ReadVariableOp(Adam/dense_14/bias/v/Read/ReadVariableOpConst*+
Tin$
"2 	*
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
GPU 2J 8� *)
f$R"
 __inference__traced_save_3439878
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decaydense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biastotalcountAdam/dense_12/kernel/mAdam/dense_12/bias/mAdam/dense_13/kernel/mAdam/dense_13/bias/mAdam/dense_14/kernel/mAdam/dense_14/bias/mAdam/dense_12/kernel/vAdam/dense_12/bias/vAdam/dense_13/kernel/vAdam/dense_13/bias/vAdam/dense_14/kernel/vAdam/dense_14/bias/v**
Tin#
!2*
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
GPU 2J 8� *,
f'R%
#__inference__traced_restore_3439978��

�
�
H__inference_pairEncoder_layer_call_and_return_conditional_losses_3438562

inputs
inputs_1,
celllineextractor_3438539:	�@'
celllineextractor_3438541:@+
celllineextractor_3438543:@@'
celllineextractor_3438545:@+
celllineextractor_3438547:@@'
celllineextractor_3438549:@
identity��)cellLineExtractor/StatefulPartitionedCall�
)cellLineExtractor/StatefulPartitionedCallStatefulPartitionedCallinputs_1celllineextractor_3438539celllineextractor_3438541celllineextractor_3438543celllineextractor_3438545celllineextractor_3438547celllineextractor_3438549*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_3438325�
concatenate_4/PartitionedCallPartitionedCallinputs2cellLineExtractor/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_concatenate_4_layer_call_and_return_conditional_losses_3438559v
IdentityIdentity&concatenate_4/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������r
NoOpNoOp*^cellLineExtractor/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:����������:����������: : : : : : 2V
)cellLineExtractor/StatefulPartitionedCall)cellLineExtractor/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_3438456

inputs 
dense_3438438:	�@
dense_3438440:@!
dense_1_3438444:@@
dense_1_3438446:@!
dense_2_3438450:@@
dense_2_3438452:@
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dropout/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3438438dense_3438440*
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
GPU 2J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_3438270�
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_3438403�
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_3438444dense_1_3438446*
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
GPU 2J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_3438294�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_3438370�
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_3438450dense_2_3438452*
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
GPU 2J 8� *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_3438318w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
e
F__inference_dropout_9_layer_call_and_return_conditional_losses_3438840

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
G
+__inference_dropout_9_layer_call_fn_3439493

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
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_9_layer_call_and_return_conditional_losses_3438763`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_3439596

inputs7
$dense_matmul_readvariableop_resource:	�@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@@5
'dense_1_biasadd_readvariableop_resource:@8
&dense_2_matmul_readvariableop_resource:@@5
'dense_2_biasadd_readvariableop_resource:@
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0u
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@h
dropout/IdentityIdentitydense/Relu:activations:0*
T0*'
_output_shapes
:���������@�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@l
dropout_1/IdentityIdentitydense_1/Relu:activations:0*
T0*'
_output_shapes
:���������@�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_2/MatMulMatMuldropout_1/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@`
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������@i
IdentityIdentitydense_2/Relu:activations:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
H__inference_pairEncoder_layer_call_and_return_conditional_losses_3438624

inputs
inputs_1,
celllineextractor_3438609:	�@'
celllineextractor_3438611:@+
celllineextractor_3438613:@@'
celllineextractor_3438615:@+
celllineextractor_3438617:@@'
celllineextractor_3438619:@
identity��)cellLineExtractor/StatefulPartitionedCall�
)cellLineExtractor/StatefulPartitionedCallStatefulPartitionedCallinputs_1celllineextractor_3438609celllineextractor_3438611celllineextractor_3438613celllineextractor_3438615celllineextractor_3438617celllineextractor_3438619*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_3438456�
concatenate_4/PartitionedCallPartitionedCallinputs2cellLineExtractor/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_concatenate_4_layer_call_and_return_conditional_losses_3438559v
IdentityIdentity&concatenate_4/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������r
NoOpNoOp*^cellLineExtractor/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:����������:����������: : : : : : 2V
)cellLineExtractor/StatefulPartitionedCall)cellLineExtractor/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
H__inference_pairEncoder_layer_call_and_return_conditional_losses_3438695
input_9
input_10,
celllineextractor_3438680:	�@'
celllineextractor_3438682:@+
celllineextractor_3438684:@@'
celllineextractor_3438686:@+
celllineextractor_3438688:@@'
celllineextractor_3438690:@
identity��)cellLineExtractor/StatefulPartitionedCall�
)cellLineExtractor/StatefulPartitionedCallStatefulPartitionedCallinput_10celllineextractor_3438680celllineextractor_3438682celllineextractor_3438684celllineextractor_3438686celllineextractor_3438688celllineextractor_3438690*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_3438456�
concatenate_4/PartitionedCallPartitionedCallinput_92cellLineExtractor/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_concatenate_4_layer_call_and_return_conditional_losses_3438559v
IdentityIdentity&concatenate_4/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������r
NoOpNoOp*^cellLineExtractor/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:����������:����������: : : : : : 2V
)cellLineExtractor/StatefulPartitionedCall)cellLineExtractor/StatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_9:RN
(
_output_shapes
:����������
"
_user_specified_name
input_10
�
b
D__inference_dropout_layer_call_and_return_conditional_losses_3439685

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
G
+__inference_dropout_1_layer_call_fn_3439722

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
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_3438305`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
B__inference_dense_layer_call_and_return_conditional_losses_3438270

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
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
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�p
�
C__inference_FS-CDR_layer_call_and_return_conditional_losses_3439279
inputs_0
inputs_1U
Bpairencoder_celllineextractor_dense_matmul_readvariableop_resource:	�@Q
Cpairencoder_celllineextractor_dense_biasadd_readvariableop_resource:@V
Dpairencoder_celllineextractor_dense_1_matmul_readvariableop_resource:@@S
Epairencoder_celllineextractor_dense_1_biasadd_readvariableop_resource:@V
Dpairencoder_celllineextractor_dense_2_matmul_readvariableop_resource:@@S
Epairencoder_celllineextractor_dense_2_biasadd_readvariableop_resource:@:
'dense_12_matmul_readvariableop_resource:	�@6
(dense_12_biasadd_readvariableop_resource:@9
'dense_13_matmul_readvariableop_resource:@@6
(dense_13_biasadd_readvariableop_resource:@9
'dense_14_matmul_readvariableop_resource:@6
(dense_14_biasadd_readvariableop_resource:
identity��dense_12/BiasAdd/ReadVariableOp�dense_12/MatMul/ReadVariableOp�dense_13/BiasAdd/ReadVariableOp�dense_13/MatMul/ReadVariableOp�dense_14/BiasAdd/ReadVariableOp�dense_14/MatMul/ReadVariableOp�:pairEncoder/cellLineExtractor/dense/BiasAdd/ReadVariableOp�9pairEncoder/cellLineExtractor/dense/MatMul/ReadVariableOp�<pairEncoder/cellLineExtractor/dense_1/BiasAdd/ReadVariableOp�;pairEncoder/cellLineExtractor/dense_1/MatMul/ReadVariableOp�<pairEncoder/cellLineExtractor/dense_2/BiasAdd/ReadVariableOp�;pairEncoder/cellLineExtractor/dense_2/MatMul/ReadVariableOp�
9pairEncoder/cellLineExtractor/dense/MatMul/ReadVariableOpReadVariableOpBpairencoder_celllineextractor_dense_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
*pairEncoder/cellLineExtractor/dense/MatMulMatMulinputs_1ApairEncoder/cellLineExtractor/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
:pairEncoder/cellLineExtractor/dense/BiasAdd/ReadVariableOpReadVariableOpCpairencoder_celllineextractor_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
+pairEncoder/cellLineExtractor/dense/BiasAddBiasAdd4pairEncoder/cellLineExtractor/dense/MatMul:product:0BpairEncoder/cellLineExtractor/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
(pairEncoder/cellLineExtractor/dense/ReluRelu4pairEncoder/cellLineExtractor/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@x
3pairEncoder/cellLineExtractor/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
1pairEncoder/cellLineExtractor/dropout/dropout/MulMul6pairEncoder/cellLineExtractor/dense/Relu:activations:0<pairEncoder/cellLineExtractor/dropout/dropout/Const:output:0*
T0*'
_output_shapes
:���������@�
3pairEncoder/cellLineExtractor/dropout/dropout/ShapeShape6pairEncoder/cellLineExtractor/dense/Relu:activations:0*
T0*
_output_shapes
:�
JpairEncoder/cellLineExtractor/dropout/dropout/random_uniform/RandomUniformRandomUniform<pairEncoder/cellLineExtractor/dropout/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0�
<pairEncoder/cellLineExtractor/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
:pairEncoder/cellLineExtractor/dropout/dropout/GreaterEqualGreaterEqualSpairEncoder/cellLineExtractor/dropout/dropout/random_uniform/RandomUniform:output:0EpairEncoder/cellLineExtractor/dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@�
2pairEncoder/cellLineExtractor/dropout/dropout/CastCast>pairEncoder/cellLineExtractor/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@�
3pairEncoder/cellLineExtractor/dropout/dropout/Mul_1Mul5pairEncoder/cellLineExtractor/dropout/dropout/Mul:z:06pairEncoder/cellLineExtractor/dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:���������@�
;pairEncoder/cellLineExtractor/dense_1/MatMul/ReadVariableOpReadVariableOpDpairencoder_celllineextractor_dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
,pairEncoder/cellLineExtractor/dense_1/MatMulMatMul7pairEncoder/cellLineExtractor/dropout/dropout/Mul_1:z:0CpairEncoder/cellLineExtractor/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
<pairEncoder/cellLineExtractor/dense_1/BiasAdd/ReadVariableOpReadVariableOpEpairencoder_celllineextractor_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
-pairEncoder/cellLineExtractor/dense_1/BiasAddBiasAdd6pairEncoder/cellLineExtractor/dense_1/MatMul:product:0DpairEncoder/cellLineExtractor/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
*pairEncoder/cellLineExtractor/dense_1/ReluRelu6pairEncoder/cellLineExtractor/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@z
5pairEncoder/cellLineExtractor/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
3pairEncoder/cellLineExtractor/dropout_1/dropout/MulMul8pairEncoder/cellLineExtractor/dense_1/Relu:activations:0>pairEncoder/cellLineExtractor/dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:���������@�
5pairEncoder/cellLineExtractor/dropout_1/dropout/ShapeShape8pairEncoder/cellLineExtractor/dense_1/Relu:activations:0*
T0*
_output_shapes
:�
LpairEncoder/cellLineExtractor/dropout_1/dropout/random_uniform/RandomUniformRandomUniform>pairEncoder/cellLineExtractor/dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0�
>pairEncoder/cellLineExtractor/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
<pairEncoder/cellLineExtractor/dropout_1/dropout/GreaterEqualGreaterEqualUpairEncoder/cellLineExtractor/dropout_1/dropout/random_uniform/RandomUniform:output:0GpairEncoder/cellLineExtractor/dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@�
4pairEncoder/cellLineExtractor/dropout_1/dropout/CastCast@pairEncoder/cellLineExtractor/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@�
5pairEncoder/cellLineExtractor/dropout_1/dropout/Mul_1Mul7pairEncoder/cellLineExtractor/dropout_1/dropout/Mul:z:08pairEncoder/cellLineExtractor/dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:���������@�
;pairEncoder/cellLineExtractor/dense_2/MatMul/ReadVariableOpReadVariableOpDpairencoder_celllineextractor_dense_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
,pairEncoder/cellLineExtractor/dense_2/MatMulMatMul9pairEncoder/cellLineExtractor/dropout_1/dropout/Mul_1:z:0CpairEncoder/cellLineExtractor/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
<pairEncoder/cellLineExtractor/dense_2/BiasAdd/ReadVariableOpReadVariableOpEpairencoder_celllineextractor_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
-pairEncoder/cellLineExtractor/dense_2/BiasAddBiasAdd6pairEncoder/cellLineExtractor/dense_2/MatMul:product:0DpairEncoder/cellLineExtractor/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
*pairEncoder/cellLineExtractor/dense_2/ReluRelu6pairEncoder/cellLineExtractor/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������@g
%pairEncoder/concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
 pairEncoder/concatenate_4/concatConcatV2inputs_08pairEncoder/cellLineExtractor/dense_2/Relu:activations:0.pairEncoder/concatenate_4/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense_12/MatMulMatMul)pairEncoder/concatenate_4/concat:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:���������@\
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_8/dropout/MulMuldense_12/Relu:activations:0 dropout_8/dropout/Const:output:0*
T0*'
_output_shapes
:���������@b
dropout_8/dropout/ShapeShapedense_12/Relu:activations:0*
T0*
_output_shapes
:�
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0e
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@�
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@�
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*'
_output_shapes
:���������@�
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_13/MatMulMatMuldropout_8/dropout/Mul_1:z:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*'
_output_shapes
:���������@\
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_9/dropout/MulMuldense_13/Relu:activations:0 dropout_9/dropout/Const:output:0*
T0*'
_output_shapes
:���������@b
dropout_9/dropout/ShapeShapedense_13/Relu:activations:0*
T0*
_output_shapes
:�
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0e
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@�
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@�
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*'
_output_shapes
:���������@�
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_14/MatMulMatMuldropout_9/dropout/Mul_1:z:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
dense_14/SigmoidSigmoiddense_14/BiasAdd:output:0*
T0*'
_output_shapes
:���������c
IdentityIdentitydense_14/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp;^pairEncoder/cellLineExtractor/dense/BiasAdd/ReadVariableOp:^pairEncoder/cellLineExtractor/dense/MatMul/ReadVariableOp=^pairEncoder/cellLineExtractor/dense_1/BiasAdd/ReadVariableOp<^pairEncoder/cellLineExtractor/dense_1/MatMul/ReadVariableOp=^pairEncoder/cellLineExtractor/dense_2/BiasAdd/ReadVariableOp<^pairEncoder/cellLineExtractor/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������:����������: : : : : : : : : : : : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2x
:pairEncoder/cellLineExtractor/dense/BiasAdd/ReadVariableOp:pairEncoder/cellLineExtractor/dense/BiasAdd/ReadVariableOp2v
9pairEncoder/cellLineExtractor/dense/MatMul/ReadVariableOp9pairEncoder/cellLineExtractor/dense/MatMul/ReadVariableOp2|
<pairEncoder/cellLineExtractor/dense_1/BiasAdd/ReadVariableOp<pairEncoder/cellLineExtractor/dense_1/BiasAdd/ReadVariableOp2z
;pairEncoder/cellLineExtractor/dense_1/MatMul/ReadVariableOp;pairEncoder/cellLineExtractor/dense_1/MatMul/ReadVariableOp2|
<pairEncoder/cellLineExtractor/dense_2/BiasAdd/ReadVariableOp<pairEncoder/cellLineExtractor/dense_2/BiasAdd/ReadVariableOp2z
;pairEncoder/cellLineExtractor/dense_2/MatMul/ReadVariableOp;pairEncoder/cellLineExtractor/dense_2/MatMul/ReadVariableOp:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1
�"
�
C__inference_FS-CDR_layer_call_and_return_conditional_losses_3439081
input_9
input_10&
pairencoder_3439050:	�@!
pairencoder_3439052:@%
pairencoder_3439054:@@!
pairencoder_3439056:@%
pairencoder_3439058:@@!
pairencoder_3439060:@#
dense_12_3439063:	�@
dense_12_3439065:@"
dense_13_3439069:@@
dense_13_3439071:@"
dense_14_3439075:@
dense_14_3439077:
identity�� dense_12/StatefulPartitionedCall� dense_13/StatefulPartitionedCall� dense_14/StatefulPartitionedCall�!dropout_8/StatefulPartitionedCall�!dropout_9/StatefulPartitionedCall�#pairEncoder/StatefulPartitionedCall�
#pairEncoder/StatefulPartitionedCallStatefulPartitionedCallinput_9input_10pairencoder_3439050pairencoder_3439052pairencoder_3439054pairencoder_3439056pairencoder_3439058pairencoder_3439060*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_pairEncoder_layer_call_and_return_conditional_losses_3438624�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall,pairEncoder/StatefulPartitionedCall:output:0dense_12_3439063dense_12_3439065*
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
GPU 2J 8� *N
fIRG
E__inference_dense_12_layer_call_and_return_conditional_losses_3438728�
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_8_layer_call_and_return_conditional_losses_3438873�
 dense_13/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0dense_13_3439069dense_13_3439071*
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
GPU 2J 8� *N
fIRG
E__inference_dense_13_layer_call_and_return_conditional_losses_3438752�
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_9_layer_call_and_return_conditional_losses_3438840�
 dense_14/StatefulPartitionedCallStatefulPartitionedCall*dropout_9/StatefulPartitionedCall:output:0dense_14_3439075dense_14_3439077*
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
GPU 2J 8� *N
fIRG
E__inference_dense_14_layer_call_and_return_conditional_losses_3438776x
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall$^pairEncoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������:����������: : : : : : : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall2J
#pairEncoder/StatefulPartitionedCall#pairEncoder/StatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_9:RN
(
_output_shapes
:����������
"
_user_specified_name
input_10
�

�
E__inference_dense_13_layer_call_and_return_conditional_losses_3438752

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
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
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
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
 
_user_specified_nameinputs
�
�
(__inference_FS-CDR_layer_call_fn_3439011
input_9
input_10
unknown:	�@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:	�@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_9input_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_FS-CDR_layer_call_and_return_conditional_losses_3438954o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������:����������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_9:RN
(
_output_shapes
:����������
"
_user_specified_name
input_10
�
�
*__inference_dense_12_layer_call_fn_3439430

inputs
unknown:	�@
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
GPU 2J 8� *N
fIRG
E__inference_dense_12_layer_call_and_return_conditional_losses_3438728o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
[
/__inference_concatenate_4_layer_call_fn_3439643
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_concatenate_4_layer_call_and_return_conditional_losses_3438559a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':����������:���������@:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������@
"
_user_specified_name
inputs/1
�
d
+__inference_dropout_1_layer_call_fn_3439727

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_3438370o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
(__inference_FS-CDR_layer_call_fn_3439115
inputs_0
inputs_1
unknown:	�@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:	�@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_FS-CDR_layer_call_and_return_conditional_losses_3438783o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������:����������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1
�
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_3439732

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_3438509
input_1 
dense_3438491:	�@
dense_3438493:@!
dense_1_3438497:@@
dense_1_3438499:@!
dense_2_3438503:@@
dense_2_3438505:@
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_3438491dense_3438493*
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
GPU 2J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_3438270�
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_3438281�
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_3438497dense_1_3438499*
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
GPU 2J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_3438294�
dropout_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_3438305�
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_2_3438503dense_2_3438505*
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
GPU 2J 8� *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_3438318w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_3438305

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
d
F__inference_dropout_9_layer_call_and_return_conditional_losses_3439503

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�?
�
 __inference__traced_save_3439878
file_prefix.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop.
*savev2_dense_14_kernel_read_readvariableop,
(savev2_dense_14_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_12_kernel_m_read_readvariableop3
/savev2_adam_dense_12_bias_m_read_readvariableop5
1savev2_adam_dense_13_kernel_m_read_readvariableop3
/savev2_adam_dense_13_bias_m_read_readvariableop5
1savev2_adam_dense_14_kernel_m_read_readvariableop3
/savev2_adam_dense_14_bias_m_read_readvariableop5
1savev2_adam_dense_12_kernel_v_read_readvariableop3
/savev2_adam_dense_12_bias_v_read_readvariableop5
1savev2_adam_dense_13_kernel_v_read_readvariableop3
/savev2_adam_dense_13_bias_v_read_readvariableop5
1savev2_adam_dense_14_kernel_v_read_readvariableop3
/savev2_adam_dense_14_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
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
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop*savev2_dense_14_kernel_read_readvariableop(savev2_dense_14_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_12_kernel_m_read_readvariableop/savev2_adam_dense_12_bias_m_read_readvariableop1savev2_adam_dense_13_kernel_m_read_readvariableop/savev2_adam_dense_13_bias_m_read_readvariableop1savev2_adam_dense_14_kernel_m_read_readvariableop/savev2_adam_dense_14_bias_m_read_readvariableop1savev2_adam_dense_12_kernel_v_read_readvariableop/savev2_adam_dense_12_bias_v_read_readvariableop1savev2_adam_dense_13_kernel_v_read_readvariableop/savev2_adam_dense_13_bias_v_read_readvariableop1savev2_adam_dense_14_kernel_v_read_readvariableop/savev2_adam_dense_14_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *-
dtypes#
!2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :	�@:@:@@:@:@:: : : : :	�@:@:@@:@:@@:@: : :	�@:@:@@:@:@::	�@:@:@@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	�@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :%!

_output_shapes
:	�@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::%!

_output_shapes
:	�@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: 
�
�
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_3438325

inputs 
dense_3438271:	�@
dense_3438273:@!
dense_1_3438295:@@
dense_1_3438297:@!
dense_2_3438319:@@
dense_2_3438321:@
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3438271dense_3438273*
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
GPU 2J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_3438270�
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_3438281�
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_3438295dense_1_3438297*
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
GPU 2J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_3438294�
dropout_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_3438305�
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_2_3438319dense_2_3438321*
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
GPU 2J 8� *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_3438318w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
)__inference_dense_2_layer_call_fn_3439753

inputs
unknown:@@
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
GPU 2J 8� *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_3438318o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
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
 
_user_specified_nameinputs
�
�
*__inference_dense_13_layer_call_fn_3439477

inputs
unknown:@@
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
GPU 2J 8� *N
fIRG
E__inference_dense_13_layer_call_and_return_conditional_losses_3438752o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
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
 
_user_specified_nameinputs
�	
c
D__inference_dropout_layer_call_and_return_conditional_losses_3438403

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
e
F__inference_dropout_1_layer_call_and_return_conditional_losses_3438370

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
d
F__inference_dropout_8_layer_call_and_return_conditional_losses_3438739

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
-__inference_pairEncoder_layer_call_fn_3438577
input_9
input_10
unknown:	�@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_9input_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_pairEncoder_layer_call_and_return_conditional_losses_3438562p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:����������:����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_9:RN
(
_output_shapes
:����������
"
_user_specified_name
input_10
�

�
E__inference_dense_13_layer_call_and_return_conditional_losses_3439488

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
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
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
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
 
_user_specified_nameinputs
�

�
D__inference_dense_2_layer_call_and_return_conditional_losses_3438318

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
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
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
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
 
_user_specified_nameinputs
�	
�
-__inference_pairEncoder_layer_call_fn_3439329
inputs_0
inputs_1
unknown:	�@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_pairEncoder_layer_call_and_return_conditional_losses_3438562p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:����������:����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1
�	
�
-__inference_pairEncoder_layer_call_fn_3438657
input_9
input_10
unknown:	�@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_9input_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_pairEncoder_layer_call_and_return_conditional_losses_3438624p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:����������:����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_9:RN
(
_output_shapes
:����������
"
_user_specified_name
input_10
�
d
+__inference_dropout_9_layer_call_fn_3439498

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_9_layer_call_and_return_conditional_losses_3438840o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
'__inference_dense_layer_call_fn_3439659

inputs
unknown:	�@
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
GPU 2J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_3438270o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_FS-CDR_layer_call_fn_3438810
input_9
input_10
unknown:	�@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:	�@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_9input_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_FS-CDR_layer_call_and_return_conditional_losses_3438783o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������:����������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_9:RN
(
_output_shapes
:����������
"
_user_specified_name
input_10
�

�
B__inference_dense_layer_call_and_return_conditional_losses_3439670

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
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
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
v
J__inference_concatenate_4_layer_call_and_return_conditional_losses_3439650
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :x
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:����������X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':����������:���������@:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������@
"
_user_specified_name
inputs/1
�
�
C__inference_FS-CDR_layer_call_and_return_conditional_losses_3439046
input_9
input_10&
pairencoder_3439015:	�@!
pairencoder_3439017:@%
pairencoder_3439019:@@!
pairencoder_3439021:@%
pairencoder_3439023:@@!
pairencoder_3439025:@#
dense_12_3439028:	�@
dense_12_3439030:@"
dense_13_3439034:@@
dense_13_3439036:@"
dense_14_3439040:@
dense_14_3439042:
identity�� dense_12/StatefulPartitionedCall� dense_13/StatefulPartitionedCall� dense_14/StatefulPartitionedCall�#pairEncoder/StatefulPartitionedCall�
#pairEncoder/StatefulPartitionedCallStatefulPartitionedCallinput_9input_10pairencoder_3439015pairencoder_3439017pairencoder_3439019pairencoder_3439021pairencoder_3439023pairencoder_3439025*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_pairEncoder_layer_call_and_return_conditional_losses_3438562�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall,pairEncoder/StatefulPartitionedCall:output:0dense_12_3439028dense_12_3439030*
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
GPU 2J 8� *N
fIRG
E__inference_dense_12_layer_call_and_return_conditional_losses_3438728�
dropout_8/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_8_layer_call_and_return_conditional_losses_3438739�
 dense_13/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0dense_13_3439034dense_13_3439036*
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
GPU 2J 8� *N
fIRG
E__inference_dense_13_layer_call_and_return_conditional_losses_3438752�
dropout_9/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_9_layer_call_and_return_conditional_losses_3438763�
 dense_14/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0dense_14_3439040dense_14_3439042*
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
GPU 2J 8� *N
fIRG
E__inference_dense_14_layer_call_and_return_conditional_losses_3438776x
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall$^pairEncoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������:����������: : : : : : : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2J
#pairEncoder/StatefulPartitionedCall#pairEncoder/StatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_9:RN
(
_output_shapes
:����������
"
_user_specified_name
input_10
�

�
D__inference_dense_1_layer_call_and_return_conditional_losses_3438294

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
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
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
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
 
_user_specified_nameinputs
�	
�
-__inference_pairEncoder_layer_call_fn_3439347
inputs_0
inputs_1
unknown:	�@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_pairEncoder_layer_call_and_return_conditional_losses_3438624p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:����������:����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1
�
E
)__inference_dropout_layer_call_fn_3439675

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
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_3438281`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
c
D__inference_dropout_layer_call_and_return_conditional_losses_3439697

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
b
)__inference_dropout_layer_call_fn_3439680

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_3438403o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
D__inference_dense_1_layer_call_and_return_conditional_losses_3439717

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
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
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
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
 
_user_specified_nameinputs
�

�
E__inference_dense_12_layer_call_and_return_conditional_losses_3438728

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
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
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
D__inference_dense_2_layer_call_and_return_conditional_losses_3439764

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
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
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
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
 
_user_specified_nameinputs
�
�
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_3438530
input_1 
dense_3438512:	�@
dense_3438514:@!
dense_1_3438518:@@
dense_1_3438520:@!
dense_2_3438524:@@
dense_2_3438526:@
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dropout/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_3438512dense_3438514*
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
GPU 2J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_3438270�
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_3438403�
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_3438518dense_1_3438520*
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
GPU 2J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_3438294�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_3438370�
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_3438524dense_2_3438526*
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
GPU 2J 8� *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_3438318w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�*
�
H__inference_pairEncoder_layer_call_and_return_conditional_losses_3439377
inputs_0
inputs_1I
6celllineextractor_dense_matmul_readvariableop_resource:	�@E
7celllineextractor_dense_biasadd_readvariableop_resource:@J
8celllineextractor_dense_1_matmul_readvariableop_resource:@@G
9celllineextractor_dense_1_biasadd_readvariableop_resource:@J
8celllineextractor_dense_2_matmul_readvariableop_resource:@@G
9celllineextractor_dense_2_biasadd_readvariableop_resource:@
identity��.cellLineExtractor/dense/BiasAdd/ReadVariableOp�-cellLineExtractor/dense/MatMul/ReadVariableOp�0cellLineExtractor/dense_1/BiasAdd/ReadVariableOp�/cellLineExtractor/dense_1/MatMul/ReadVariableOp�0cellLineExtractor/dense_2/BiasAdd/ReadVariableOp�/cellLineExtractor/dense_2/MatMul/ReadVariableOp�
-cellLineExtractor/dense/MatMul/ReadVariableOpReadVariableOp6celllineextractor_dense_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
cellLineExtractor/dense/MatMulMatMulinputs_15cellLineExtractor/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
.cellLineExtractor/dense/BiasAdd/ReadVariableOpReadVariableOp7celllineextractor_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
cellLineExtractor/dense/BiasAddBiasAdd(cellLineExtractor/dense/MatMul:product:06cellLineExtractor/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
cellLineExtractor/dense/ReluRelu(cellLineExtractor/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
"cellLineExtractor/dropout/IdentityIdentity*cellLineExtractor/dense/Relu:activations:0*
T0*'
_output_shapes
:���������@�
/cellLineExtractor/dense_1/MatMul/ReadVariableOpReadVariableOp8celllineextractor_dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
 cellLineExtractor/dense_1/MatMulMatMul+cellLineExtractor/dropout/Identity:output:07cellLineExtractor/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
0cellLineExtractor/dense_1/BiasAdd/ReadVariableOpReadVariableOp9celllineextractor_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
!cellLineExtractor/dense_1/BiasAddBiasAdd*cellLineExtractor/dense_1/MatMul:product:08cellLineExtractor/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
cellLineExtractor/dense_1/ReluRelu*cellLineExtractor/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
$cellLineExtractor/dropout_1/IdentityIdentity,cellLineExtractor/dense_1/Relu:activations:0*
T0*'
_output_shapes
:���������@�
/cellLineExtractor/dense_2/MatMul/ReadVariableOpReadVariableOp8celllineextractor_dense_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
 cellLineExtractor/dense_2/MatMulMatMul-cellLineExtractor/dropout_1/Identity:output:07cellLineExtractor/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
0cellLineExtractor/dense_2/BiasAdd/ReadVariableOpReadVariableOp9celllineextractor_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
!cellLineExtractor/dense_2/BiasAddBiasAdd*cellLineExtractor/dense_2/MatMul:product:08cellLineExtractor/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
cellLineExtractor/dense_2/ReluRelu*cellLineExtractor/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������@[
concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_4/concatConcatV2inputs_0,cellLineExtractor/dense_2/Relu:activations:0"concatenate_4/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������m
IdentityIdentityconcatenate_4/concat:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp/^cellLineExtractor/dense/BiasAdd/ReadVariableOp.^cellLineExtractor/dense/MatMul/ReadVariableOp1^cellLineExtractor/dense_1/BiasAdd/ReadVariableOp0^cellLineExtractor/dense_1/MatMul/ReadVariableOp1^cellLineExtractor/dense_2/BiasAdd/ReadVariableOp0^cellLineExtractor/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:����������:����������: : : : : : 2`
.cellLineExtractor/dense/BiasAdd/ReadVariableOp.cellLineExtractor/dense/BiasAdd/ReadVariableOp2^
-cellLineExtractor/dense/MatMul/ReadVariableOp-cellLineExtractor/dense/MatMul/ReadVariableOp2d
0cellLineExtractor/dense_1/BiasAdd/ReadVariableOp0cellLineExtractor/dense_1/BiasAdd/ReadVariableOp2b
/cellLineExtractor/dense_1/MatMul/ReadVariableOp/cellLineExtractor/dense_1/MatMul/ReadVariableOp2d
0cellLineExtractor/dense_2/BiasAdd/ReadVariableOp0cellLineExtractor/dense_2/BiasAdd/ReadVariableOp2b
/cellLineExtractor/dense_2/MatMul/ReadVariableOp/cellLineExtractor/dense_2/MatMul/ReadVariableOp:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1
�

�
E__inference_dense_14_layer_call_and_return_conditional_losses_3438776

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
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
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
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
 
_user_specified_nameinputs
�"
�
C__inference_FS-CDR_layer_call_and_return_conditional_losses_3438954

inputs
inputs_1&
pairencoder_3438923:	�@!
pairencoder_3438925:@%
pairencoder_3438927:@@!
pairencoder_3438929:@%
pairencoder_3438931:@@!
pairencoder_3438933:@#
dense_12_3438936:	�@
dense_12_3438938:@"
dense_13_3438942:@@
dense_13_3438944:@"
dense_14_3438948:@
dense_14_3438950:
identity�� dense_12/StatefulPartitionedCall� dense_13/StatefulPartitionedCall� dense_14/StatefulPartitionedCall�!dropout_8/StatefulPartitionedCall�!dropout_9/StatefulPartitionedCall�#pairEncoder/StatefulPartitionedCall�
#pairEncoder/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1pairencoder_3438923pairencoder_3438925pairencoder_3438927pairencoder_3438929pairencoder_3438931pairencoder_3438933*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_pairEncoder_layer_call_and_return_conditional_losses_3438624�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall,pairEncoder/StatefulPartitionedCall:output:0dense_12_3438936dense_12_3438938*
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
GPU 2J 8� *N
fIRG
E__inference_dense_12_layer_call_and_return_conditional_losses_3438728�
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_8_layer_call_and_return_conditional_losses_3438873�
 dense_13/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0dense_13_3438942dense_13_3438944*
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
GPU 2J 8� *N
fIRG
E__inference_dense_13_layer_call_and_return_conditional_losses_3438752�
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_9_layer_call_and_return_conditional_losses_3438840�
 dense_14/StatefulPartitionedCallStatefulPartitionedCall*dropout_9/StatefulPartitionedCall:output:0dense_14_3438948dense_14_3438950*
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
GPU 2J 8� *N
fIRG
E__inference_dense_14_layer_call_and_return_conditional_losses_3438776x
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall$^pairEncoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������:����������: : : : : : : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall2J
#pairEncoder/StatefulPartitionedCall#pairEncoder/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
d
+__inference_dropout_8_layer_call_fn_3439451

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_8_layer_call_and_return_conditional_losses_3438873o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
E__inference_dense_12_layer_call_and_return_conditional_losses_3439441

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
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
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�K
�
C__inference_FS-CDR_layer_call_and_return_conditional_losses_3439198
inputs_0
inputs_1U
Bpairencoder_celllineextractor_dense_matmul_readvariableop_resource:	�@Q
Cpairencoder_celllineextractor_dense_biasadd_readvariableop_resource:@V
Dpairencoder_celllineextractor_dense_1_matmul_readvariableop_resource:@@S
Epairencoder_celllineextractor_dense_1_biasadd_readvariableop_resource:@V
Dpairencoder_celllineextractor_dense_2_matmul_readvariableop_resource:@@S
Epairencoder_celllineextractor_dense_2_biasadd_readvariableop_resource:@:
'dense_12_matmul_readvariableop_resource:	�@6
(dense_12_biasadd_readvariableop_resource:@9
'dense_13_matmul_readvariableop_resource:@@6
(dense_13_biasadd_readvariableop_resource:@9
'dense_14_matmul_readvariableop_resource:@6
(dense_14_biasadd_readvariableop_resource:
identity��dense_12/BiasAdd/ReadVariableOp�dense_12/MatMul/ReadVariableOp�dense_13/BiasAdd/ReadVariableOp�dense_13/MatMul/ReadVariableOp�dense_14/BiasAdd/ReadVariableOp�dense_14/MatMul/ReadVariableOp�:pairEncoder/cellLineExtractor/dense/BiasAdd/ReadVariableOp�9pairEncoder/cellLineExtractor/dense/MatMul/ReadVariableOp�<pairEncoder/cellLineExtractor/dense_1/BiasAdd/ReadVariableOp�;pairEncoder/cellLineExtractor/dense_1/MatMul/ReadVariableOp�<pairEncoder/cellLineExtractor/dense_2/BiasAdd/ReadVariableOp�;pairEncoder/cellLineExtractor/dense_2/MatMul/ReadVariableOp�
9pairEncoder/cellLineExtractor/dense/MatMul/ReadVariableOpReadVariableOpBpairencoder_celllineextractor_dense_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
*pairEncoder/cellLineExtractor/dense/MatMulMatMulinputs_1ApairEncoder/cellLineExtractor/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
:pairEncoder/cellLineExtractor/dense/BiasAdd/ReadVariableOpReadVariableOpCpairencoder_celllineextractor_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
+pairEncoder/cellLineExtractor/dense/BiasAddBiasAdd4pairEncoder/cellLineExtractor/dense/MatMul:product:0BpairEncoder/cellLineExtractor/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
(pairEncoder/cellLineExtractor/dense/ReluRelu4pairEncoder/cellLineExtractor/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
.pairEncoder/cellLineExtractor/dropout/IdentityIdentity6pairEncoder/cellLineExtractor/dense/Relu:activations:0*
T0*'
_output_shapes
:���������@�
;pairEncoder/cellLineExtractor/dense_1/MatMul/ReadVariableOpReadVariableOpDpairencoder_celllineextractor_dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
,pairEncoder/cellLineExtractor/dense_1/MatMulMatMul7pairEncoder/cellLineExtractor/dropout/Identity:output:0CpairEncoder/cellLineExtractor/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
<pairEncoder/cellLineExtractor/dense_1/BiasAdd/ReadVariableOpReadVariableOpEpairencoder_celllineextractor_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
-pairEncoder/cellLineExtractor/dense_1/BiasAddBiasAdd6pairEncoder/cellLineExtractor/dense_1/MatMul:product:0DpairEncoder/cellLineExtractor/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
*pairEncoder/cellLineExtractor/dense_1/ReluRelu6pairEncoder/cellLineExtractor/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
0pairEncoder/cellLineExtractor/dropout_1/IdentityIdentity8pairEncoder/cellLineExtractor/dense_1/Relu:activations:0*
T0*'
_output_shapes
:���������@�
;pairEncoder/cellLineExtractor/dense_2/MatMul/ReadVariableOpReadVariableOpDpairencoder_celllineextractor_dense_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
,pairEncoder/cellLineExtractor/dense_2/MatMulMatMul9pairEncoder/cellLineExtractor/dropout_1/Identity:output:0CpairEncoder/cellLineExtractor/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
<pairEncoder/cellLineExtractor/dense_2/BiasAdd/ReadVariableOpReadVariableOpEpairencoder_celllineextractor_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
-pairEncoder/cellLineExtractor/dense_2/BiasAddBiasAdd6pairEncoder/cellLineExtractor/dense_2/MatMul:product:0DpairEncoder/cellLineExtractor/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
*pairEncoder/cellLineExtractor/dense_2/ReluRelu6pairEncoder/cellLineExtractor/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������@g
%pairEncoder/concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
 pairEncoder/concatenate_4/concatConcatV2inputs_08pairEncoder/cellLineExtractor/dense_2/Relu:activations:0.pairEncoder/concatenate_4/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense_12/MatMulMatMul)pairEncoder/concatenate_4/concat:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:���������@m
dropout_8/IdentityIdentitydense_12/Relu:activations:0*
T0*'
_output_shapes
:���������@�
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_13/MatMulMatMuldropout_8/Identity:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*'
_output_shapes
:���������@m
dropout_9/IdentityIdentitydense_13/Relu:activations:0*
T0*'
_output_shapes
:���������@�
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_14/MatMulMatMuldropout_9/Identity:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
dense_14/SigmoidSigmoiddense_14/BiasAdd:output:0*
T0*'
_output_shapes
:���������c
IdentityIdentitydense_14/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp;^pairEncoder/cellLineExtractor/dense/BiasAdd/ReadVariableOp:^pairEncoder/cellLineExtractor/dense/MatMul/ReadVariableOp=^pairEncoder/cellLineExtractor/dense_1/BiasAdd/ReadVariableOp<^pairEncoder/cellLineExtractor/dense_1/MatMul/ReadVariableOp=^pairEncoder/cellLineExtractor/dense_2/BiasAdd/ReadVariableOp<^pairEncoder/cellLineExtractor/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������:����������: : : : : : : : : : : : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2x
:pairEncoder/cellLineExtractor/dense/BiasAdd/ReadVariableOp:pairEncoder/cellLineExtractor/dense/BiasAdd/ReadVariableOp2v
9pairEncoder/cellLineExtractor/dense/MatMul/ReadVariableOp9pairEncoder/cellLineExtractor/dense/MatMul/ReadVariableOp2|
<pairEncoder/cellLineExtractor/dense_1/BiasAdd/ReadVariableOp<pairEncoder/cellLineExtractor/dense_1/BiasAdd/ReadVariableOp2z
;pairEncoder/cellLineExtractor/dense_1/MatMul/ReadVariableOp;pairEncoder/cellLineExtractor/dense_1/MatMul/ReadVariableOp2|
<pairEncoder/cellLineExtractor/dense_2/BiasAdd/ReadVariableOp<pairEncoder/cellLineExtractor/dense_2/BiasAdd/ReadVariableOp2z
;pairEncoder/cellLineExtractor/dense_2/MatMul/ReadVariableOp;pairEncoder/cellLineExtractor/dense_2/MatMul/ReadVariableOp:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1
�	
�
3__inference_cellLineExtractor_layer_call_fn_3439569

inputs
unknown:	�@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_3438456o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
3__inference_cellLineExtractor_layer_call_fn_3439552

inputs
unknown:	�@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_3438325o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
3__inference_cellLineExtractor_layer_call_fn_3438488
input_1
unknown:	�@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_3438456o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�w
�
#__inference__traced_restore_3439978
file_prefix3
 assignvariableop_dense_12_kernel:	�@.
 assignvariableop_1_dense_12_bias:@4
"assignvariableop_2_dense_13_kernel:@@.
 assignvariableop_3_dense_13_bias:@4
"assignvariableop_4_dense_14_kernel:@.
 assignvariableop_5_dense_14_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 3
 assignvariableop_10_dense_kernel:	�@,
assignvariableop_11_dense_bias:@4
"assignvariableop_12_dense_1_kernel:@@.
 assignvariableop_13_dense_1_bias:@4
"assignvariableop_14_dense_2_kernel:@@.
 assignvariableop_15_dense_2_bias:@#
assignvariableop_16_total: #
assignvariableop_17_count: =
*assignvariableop_18_adam_dense_12_kernel_m:	�@6
(assignvariableop_19_adam_dense_12_bias_m:@<
*assignvariableop_20_adam_dense_13_kernel_m:@@6
(assignvariableop_21_adam_dense_13_bias_m:@<
*assignvariableop_22_adam_dense_14_kernel_m:@6
(assignvariableop_23_adam_dense_14_bias_m:=
*assignvariableop_24_adam_dense_12_kernel_v:	�@6
(assignvariableop_25_adam_dense_12_bias_v:@<
*assignvariableop_26_adam_dense_13_kernel_v:@@6
(assignvariableop_27_adam_dense_13_bias_v:@<
*assignvariableop_28_adam_dense_14_kernel_v:@6
(assignvariableop_29_adam_dense_14_bias_v:
identity_31��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes~
|:::::::::::::::::::::::::::::::*-
dtypes#
!2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_dense_12_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_12_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_13_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_13_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_14_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_14_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp assignvariableop_10_dense_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_dense_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_2_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_2_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_dense_12_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_dense_12_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_dense_13_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_dense_13_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_dense_14_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_dense_14_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_dense_12_kernel_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_dense_12_bias_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_dense_13_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_dense_13_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_dense_14_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_dense_14_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_30Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_31IdentityIdentity_30:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_31Identity_31:output:0*Q
_input_shapes@
>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

�
E__inference_dense_14_layer_call_and_return_conditional_losses_3439535

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
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
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
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
 
_user_specified_nameinputs
�=
�
H__inference_pairEncoder_layer_call_and_return_conditional_losses_3439421
inputs_0
inputs_1I
6celllineextractor_dense_matmul_readvariableop_resource:	�@E
7celllineextractor_dense_biasadd_readvariableop_resource:@J
8celllineextractor_dense_1_matmul_readvariableop_resource:@@G
9celllineextractor_dense_1_biasadd_readvariableop_resource:@J
8celllineextractor_dense_2_matmul_readvariableop_resource:@@G
9celllineextractor_dense_2_biasadd_readvariableop_resource:@
identity��.cellLineExtractor/dense/BiasAdd/ReadVariableOp�-cellLineExtractor/dense/MatMul/ReadVariableOp�0cellLineExtractor/dense_1/BiasAdd/ReadVariableOp�/cellLineExtractor/dense_1/MatMul/ReadVariableOp�0cellLineExtractor/dense_2/BiasAdd/ReadVariableOp�/cellLineExtractor/dense_2/MatMul/ReadVariableOp�
-cellLineExtractor/dense/MatMul/ReadVariableOpReadVariableOp6celllineextractor_dense_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
cellLineExtractor/dense/MatMulMatMulinputs_15cellLineExtractor/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
.cellLineExtractor/dense/BiasAdd/ReadVariableOpReadVariableOp7celllineextractor_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
cellLineExtractor/dense/BiasAddBiasAdd(cellLineExtractor/dense/MatMul:product:06cellLineExtractor/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
cellLineExtractor/dense/ReluRelu(cellLineExtractor/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@l
'cellLineExtractor/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
%cellLineExtractor/dropout/dropout/MulMul*cellLineExtractor/dense/Relu:activations:00cellLineExtractor/dropout/dropout/Const:output:0*
T0*'
_output_shapes
:���������@�
'cellLineExtractor/dropout/dropout/ShapeShape*cellLineExtractor/dense/Relu:activations:0*
T0*
_output_shapes
:�
>cellLineExtractor/dropout/dropout/random_uniform/RandomUniformRandomUniform0cellLineExtractor/dropout/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0u
0cellLineExtractor/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
.cellLineExtractor/dropout/dropout/GreaterEqualGreaterEqualGcellLineExtractor/dropout/dropout/random_uniform/RandomUniform:output:09cellLineExtractor/dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@�
&cellLineExtractor/dropout/dropout/CastCast2cellLineExtractor/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@�
'cellLineExtractor/dropout/dropout/Mul_1Mul)cellLineExtractor/dropout/dropout/Mul:z:0*cellLineExtractor/dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:���������@�
/cellLineExtractor/dense_1/MatMul/ReadVariableOpReadVariableOp8celllineextractor_dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
 cellLineExtractor/dense_1/MatMulMatMul+cellLineExtractor/dropout/dropout/Mul_1:z:07cellLineExtractor/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
0cellLineExtractor/dense_1/BiasAdd/ReadVariableOpReadVariableOp9celllineextractor_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
!cellLineExtractor/dense_1/BiasAddBiasAdd*cellLineExtractor/dense_1/MatMul:product:08cellLineExtractor/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
cellLineExtractor/dense_1/ReluRelu*cellLineExtractor/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@n
)cellLineExtractor/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
'cellLineExtractor/dropout_1/dropout/MulMul,cellLineExtractor/dense_1/Relu:activations:02cellLineExtractor/dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:���������@�
)cellLineExtractor/dropout_1/dropout/ShapeShape,cellLineExtractor/dense_1/Relu:activations:0*
T0*
_output_shapes
:�
@cellLineExtractor/dropout_1/dropout/random_uniform/RandomUniformRandomUniform2cellLineExtractor/dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0w
2cellLineExtractor/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
0cellLineExtractor/dropout_1/dropout/GreaterEqualGreaterEqualIcellLineExtractor/dropout_1/dropout/random_uniform/RandomUniform:output:0;cellLineExtractor/dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@�
(cellLineExtractor/dropout_1/dropout/CastCast4cellLineExtractor/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@�
)cellLineExtractor/dropout_1/dropout/Mul_1Mul+cellLineExtractor/dropout_1/dropout/Mul:z:0,cellLineExtractor/dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:���������@�
/cellLineExtractor/dense_2/MatMul/ReadVariableOpReadVariableOp8celllineextractor_dense_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
 cellLineExtractor/dense_2/MatMulMatMul-cellLineExtractor/dropout_1/dropout/Mul_1:z:07cellLineExtractor/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
0cellLineExtractor/dense_2/BiasAdd/ReadVariableOpReadVariableOp9celllineextractor_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
!cellLineExtractor/dense_2/BiasAddBiasAdd*cellLineExtractor/dense_2/MatMul:product:08cellLineExtractor/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
cellLineExtractor/dense_2/ReluRelu*cellLineExtractor/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������@[
concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_4/concatConcatV2inputs_0,cellLineExtractor/dense_2/Relu:activations:0"concatenate_4/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������m
IdentityIdentityconcatenate_4/concat:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp/^cellLineExtractor/dense/BiasAdd/ReadVariableOp.^cellLineExtractor/dense/MatMul/ReadVariableOp1^cellLineExtractor/dense_1/BiasAdd/ReadVariableOp0^cellLineExtractor/dense_1/MatMul/ReadVariableOp1^cellLineExtractor/dense_2/BiasAdd/ReadVariableOp0^cellLineExtractor/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:����������:����������: : : : : : 2`
.cellLineExtractor/dense/BiasAdd/ReadVariableOp.cellLineExtractor/dense/BiasAdd/ReadVariableOp2^
-cellLineExtractor/dense/MatMul/ReadVariableOp-cellLineExtractor/dense/MatMul/ReadVariableOp2d
0cellLineExtractor/dense_1/BiasAdd/ReadVariableOp0cellLineExtractor/dense_1/BiasAdd/ReadVariableOp2b
/cellLineExtractor/dense_1/MatMul/ReadVariableOp/cellLineExtractor/dense_1/MatMul/ReadVariableOp2d
0cellLineExtractor/dense_2/BiasAdd/ReadVariableOp0cellLineExtractor/dense_2/BiasAdd/ReadVariableOp2b
/cellLineExtractor/dense_2/MatMul/ReadVariableOp/cellLineExtractor/dense_2/MatMul/ReadVariableOp:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1
�
�
H__inference_pairEncoder_layer_call_and_return_conditional_losses_3438676
input_9
input_10,
celllineextractor_3438661:	�@'
celllineextractor_3438663:@+
celllineextractor_3438665:@@'
celllineextractor_3438667:@+
celllineextractor_3438669:@@'
celllineextractor_3438671:@
identity��)cellLineExtractor/StatefulPartitionedCall�
)cellLineExtractor/StatefulPartitionedCallStatefulPartitionedCallinput_10celllineextractor_3438661celllineextractor_3438663celllineextractor_3438665celllineextractor_3438667celllineextractor_3438669celllineextractor_3438671*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_3438325�
concatenate_4/PartitionedCallPartitionedCallinput_92cellLineExtractor/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_concatenate_4_layer_call_and_return_conditional_losses_3438559v
IdentityIdentity&concatenate_4/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������r
NoOpNoOp*^cellLineExtractor/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:����������:����������: : : : : : 2V
)cellLineExtractor/StatefulPartitionedCall)cellLineExtractor/StatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_9:RN
(
_output_shapes
:����������
"
_user_specified_name
input_10
�
�
*__inference_dense_14_layer_call_fn_3439524

inputs
unknown:@
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
GPU 2J 8� *N
fIRG
E__inference_dense_14_layer_call_and_return_conditional_losses_3438776o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
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
 
_user_specified_nameinputs
�	
�
3__inference_cellLineExtractor_layer_call_fn_3438340
input_1
unknown:	�@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_3438325o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�
d
F__inference_dropout_8_layer_call_and_return_conditional_losses_3439456

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
e
F__inference_dropout_9_layer_call_and_return_conditional_losses_3439515

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
b
D__inference_dropout_layer_call_and_return_conditional_losses_3438281

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
t
J__inference_concatenate_4_layer_call_and_return_conditional_losses_3438559

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :v
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:����������X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':����������:���������@:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
%__inference_signature_wrapper_3439311
input_10
input_9
unknown:	�@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:	�@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_9input_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_3438252o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������:����������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:����������
"
_user_specified_name
input_10:QM
(
_output_shapes
:����������
!
_user_specified_name	input_9
�	
e
F__inference_dropout_8_layer_call_and_return_conditional_losses_3438873

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�,
�
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_3439637

inputs7
$dense_matmul_readvariableop_resource:	�@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@@5
'dense_1_biasadd_readvariableop_resource:@8
&dense_2_matmul_readvariableop_resource:@@5
'dense_2_biasadd_readvariableop_resource:@
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0u
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout/dropout/MulMuldense/Relu:activations:0dropout/dropout/Const:output:0*
T0*'
_output_shapes
:���������@]
dropout/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:�
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@�
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:���������@�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_1/dropout/MulMuldense_1/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:���������@a
dropout_1/dropout/ShapeShapedense_1/Relu:activations:0*
T0*
_output_shapes
:�
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@�
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@�
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:���������@�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_2/MatMulMatMuldropout_1/dropout/Mul_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@`
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������@i
IdentityIdentitydense_2/Relu:activations:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
e
F__inference_dropout_1_layer_call_and_return_conditional_losses_3439744

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�S
�
"__inference__wrapped_model_3438252
input_9
input_10\
Ifs_cdr_pairencoder_celllineextractor_dense_matmul_readvariableop_resource:	�@X
Jfs_cdr_pairencoder_celllineextractor_dense_biasadd_readvariableop_resource:@]
Kfs_cdr_pairencoder_celllineextractor_dense_1_matmul_readvariableop_resource:@@Z
Lfs_cdr_pairencoder_celllineextractor_dense_1_biasadd_readvariableop_resource:@]
Kfs_cdr_pairencoder_celllineextractor_dense_2_matmul_readvariableop_resource:@@Z
Lfs_cdr_pairencoder_celllineextractor_dense_2_biasadd_readvariableop_resource:@A
.fs_cdr_dense_12_matmul_readvariableop_resource:	�@=
/fs_cdr_dense_12_biasadd_readvariableop_resource:@@
.fs_cdr_dense_13_matmul_readvariableop_resource:@@=
/fs_cdr_dense_13_biasadd_readvariableop_resource:@@
.fs_cdr_dense_14_matmul_readvariableop_resource:@=
/fs_cdr_dense_14_biasadd_readvariableop_resource:
identity��&FS-CDR/dense_12/BiasAdd/ReadVariableOp�%FS-CDR/dense_12/MatMul/ReadVariableOp�&FS-CDR/dense_13/BiasAdd/ReadVariableOp�%FS-CDR/dense_13/MatMul/ReadVariableOp�&FS-CDR/dense_14/BiasAdd/ReadVariableOp�%FS-CDR/dense_14/MatMul/ReadVariableOp�AFS-CDR/pairEncoder/cellLineExtractor/dense/BiasAdd/ReadVariableOp�@FS-CDR/pairEncoder/cellLineExtractor/dense/MatMul/ReadVariableOp�CFS-CDR/pairEncoder/cellLineExtractor/dense_1/BiasAdd/ReadVariableOp�BFS-CDR/pairEncoder/cellLineExtractor/dense_1/MatMul/ReadVariableOp�CFS-CDR/pairEncoder/cellLineExtractor/dense_2/BiasAdd/ReadVariableOp�BFS-CDR/pairEncoder/cellLineExtractor/dense_2/MatMul/ReadVariableOp�
@FS-CDR/pairEncoder/cellLineExtractor/dense/MatMul/ReadVariableOpReadVariableOpIfs_cdr_pairencoder_celllineextractor_dense_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
1FS-CDR/pairEncoder/cellLineExtractor/dense/MatMulMatMulinput_10HFS-CDR/pairEncoder/cellLineExtractor/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
AFS-CDR/pairEncoder/cellLineExtractor/dense/BiasAdd/ReadVariableOpReadVariableOpJfs_cdr_pairencoder_celllineextractor_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
2FS-CDR/pairEncoder/cellLineExtractor/dense/BiasAddBiasAdd;FS-CDR/pairEncoder/cellLineExtractor/dense/MatMul:product:0IFS-CDR/pairEncoder/cellLineExtractor/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
/FS-CDR/pairEncoder/cellLineExtractor/dense/ReluRelu;FS-CDR/pairEncoder/cellLineExtractor/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
5FS-CDR/pairEncoder/cellLineExtractor/dropout/IdentityIdentity=FS-CDR/pairEncoder/cellLineExtractor/dense/Relu:activations:0*
T0*'
_output_shapes
:���������@�
BFS-CDR/pairEncoder/cellLineExtractor/dense_1/MatMul/ReadVariableOpReadVariableOpKfs_cdr_pairencoder_celllineextractor_dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
3FS-CDR/pairEncoder/cellLineExtractor/dense_1/MatMulMatMul>FS-CDR/pairEncoder/cellLineExtractor/dropout/Identity:output:0JFS-CDR/pairEncoder/cellLineExtractor/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
CFS-CDR/pairEncoder/cellLineExtractor/dense_1/BiasAdd/ReadVariableOpReadVariableOpLfs_cdr_pairencoder_celllineextractor_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
4FS-CDR/pairEncoder/cellLineExtractor/dense_1/BiasAddBiasAdd=FS-CDR/pairEncoder/cellLineExtractor/dense_1/MatMul:product:0KFS-CDR/pairEncoder/cellLineExtractor/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
1FS-CDR/pairEncoder/cellLineExtractor/dense_1/ReluRelu=FS-CDR/pairEncoder/cellLineExtractor/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
7FS-CDR/pairEncoder/cellLineExtractor/dropout_1/IdentityIdentity?FS-CDR/pairEncoder/cellLineExtractor/dense_1/Relu:activations:0*
T0*'
_output_shapes
:���������@�
BFS-CDR/pairEncoder/cellLineExtractor/dense_2/MatMul/ReadVariableOpReadVariableOpKfs_cdr_pairencoder_celllineextractor_dense_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
3FS-CDR/pairEncoder/cellLineExtractor/dense_2/MatMulMatMul@FS-CDR/pairEncoder/cellLineExtractor/dropout_1/Identity:output:0JFS-CDR/pairEncoder/cellLineExtractor/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
CFS-CDR/pairEncoder/cellLineExtractor/dense_2/BiasAdd/ReadVariableOpReadVariableOpLfs_cdr_pairencoder_celllineextractor_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
4FS-CDR/pairEncoder/cellLineExtractor/dense_2/BiasAddBiasAdd=FS-CDR/pairEncoder/cellLineExtractor/dense_2/MatMul:product:0KFS-CDR/pairEncoder/cellLineExtractor/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
1FS-CDR/pairEncoder/cellLineExtractor/dense_2/ReluRelu=FS-CDR/pairEncoder/cellLineExtractor/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������@n
,FS-CDR/pairEncoder/concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
'FS-CDR/pairEncoder/concatenate_4/concatConcatV2input_9?FS-CDR/pairEncoder/cellLineExtractor/dense_2/Relu:activations:05FS-CDR/pairEncoder/concatenate_4/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
%FS-CDR/dense_12/MatMul/ReadVariableOpReadVariableOp.fs_cdr_dense_12_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
FS-CDR/dense_12/MatMulMatMul0FS-CDR/pairEncoder/concatenate_4/concat:output:0-FS-CDR/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
&FS-CDR/dense_12/BiasAdd/ReadVariableOpReadVariableOp/fs_cdr_dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
FS-CDR/dense_12/BiasAddBiasAdd FS-CDR/dense_12/MatMul:product:0.FS-CDR/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@p
FS-CDR/dense_12/ReluRelu FS-CDR/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:���������@{
FS-CDR/dropout_8/IdentityIdentity"FS-CDR/dense_12/Relu:activations:0*
T0*'
_output_shapes
:���������@�
%FS-CDR/dense_13/MatMul/ReadVariableOpReadVariableOp.fs_cdr_dense_13_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
FS-CDR/dense_13/MatMulMatMul"FS-CDR/dropout_8/Identity:output:0-FS-CDR/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
&FS-CDR/dense_13/BiasAdd/ReadVariableOpReadVariableOp/fs_cdr_dense_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
FS-CDR/dense_13/BiasAddBiasAdd FS-CDR/dense_13/MatMul:product:0.FS-CDR/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@p
FS-CDR/dense_13/ReluRelu FS-CDR/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:���������@{
FS-CDR/dropout_9/IdentityIdentity"FS-CDR/dense_13/Relu:activations:0*
T0*'
_output_shapes
:���������@�
%FS-CDR/dense_14/MatMul/ReadVariableOpReadVariableOp.fs_cdr_dense_14_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
FS-CDR/dense_14/MatMulMatMul"FS-CDR/dropout_9/Identity:output:0-FS-CDR/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
&FS-CDR/dense_14/BiasAdd/ReadVariableOpReadVariableOp/fs_cdr_dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
FS-CDR/dense_14/BiasAddBiasAdd FS-CDR/dense_14/MatMul:product:0.FS-CDR/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
FS-CDR/dense_14/SigmoidSigmoid FS-CDR/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:���������j
IdentityIdentityFS-CDR/dense_14/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp'^FS-CDR/dense_12/BiasAdd/ReadVariableOp&^FS-CDR/dense_12/MatMul/ReadVariableOp'^FS-CDR/dense_13/BiasAdd/ReadVariableOp&^FS-CDR/dense_13/MatMul/ReadVariableOp'^FS-CDR/dense_14/BiasAdd/ReadVariableOp&^FS-CDR/dense_14/MatMul/ReadVariableOpB^FS-CDR/pairEncoder/cellLineExtractor/dense/BiasAdd/ReadVariableOpA^FS-CDR/pairEncoder/cellLineExtractor/dense/MatMul/ReadVariableOpD^FS-CDR/pairEncoder/cellLineExtractor/dense_1/BiasAdd/ReadVariableOpC^FS-CDR/pairEncoder/cellLineExtractor/dense_1/MatMul/ReadVariableOpD^FS-CDR/pairEncoder/cellLineExtractor/dense_2/BiasAdd/ReadVariableOpC^FS-CDR/pairEncoder/cellLineExtractor/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������:����������: : : : : : : : : : : : 2P
&FS-CDR/dense_12/BiasAdd/ReadVariableOp&FS-CDR/dense_12/BiasAdd/ReadVariableOp2N
%FS-CDR/dense_12/MatMul/ReadVariableOp%FS-CDR/dense_12/MatMul/ReadVariableOp2P
&FS-CDR/dense_13/BiasAdd/ReadVariableOp&FS-CDR/dense_13/BiasAdd/ReadVariableOp2N
%FS-CDR/dense_13/MatMul/ReadVariableOp%FS-CDR/dense_13/MatMul/ReadVariableOp2P
&FS-CDR/dense_14/BiasAdd/ReadVariableOp&FS-CDR/dense_14/BiasAdd/ReadVariableOp2N
%FS-CDR/dense_14/MatMul/ReadVariableOp%FS-CDR/dense_14/MatMul/ReadVariableOp2�
AFS-CDR/pairEncoder/cellLineExtractor/dense/BiasAdd/ReadVariableOpAFS-CDR/pairEncoder/cellLineExtractor/dense/BiasAdd/ReadVariableOp2�
@FS-CDR/pairEncoder/cellLineExtractor/dense/MatMul/ReadVariableOp@FS-CDR/pairEncoder/cellLineExtractor/dense/MatMul/ReadVariableOp2�
CFS-CDR/pairEncoder/cellLineExtractor/dense_1/BiasAdd/ReadVariableOpCFS-CDR/pairEncoder/cellLineExtractor/dense_1/BiasAdd/ReadVariableOp2�
BFS-CDR/pairEncoder/cellLineExtractor/dense_1/MatMul/ReadVariableOpBFS-CDR/pairEncoder/cellLineExtractor/dense_1/MatMul/ReadVariableOp2�
CFS-CDR/pairEncoder/cellLineExtractor/dense_2/BiasAdd/ReadVariableOpCFS-CDR/pairEncoder/cellLineExtractor/dense_2/BiasAdd/ReadVariableOp2�
BFS-CDR/pairEncoder/cellLineExtractor/dense_2/MatMul/ReadVariableOpBFS-CDR/pairEncoder/cellLineExtractor/dense_2/MatMul/ReadVariableOp:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_9:RN
(
_output_shapes
:����������
"
_user_specified_name
input_10
�	
e
F__inference_dropout_8_layer_call_and_return_conditional_losses_3439468

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
)__inference_dense_1_layer_call_fn_3439706

inputs
unknown:@@
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
GPU 2J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_3438294o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
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
 
_user_specified_nameinputs
�
G
+__inference_dropout_8_layer_call_fn_3439446

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
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_8_layer_call_and_return_conditional_losses_3438739`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
(__inference_FS-CDR_layer_call_fn_3439145
inputs_0
inputs_1
unknown:	�@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:	�@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_FS-CDR_layer_call_and_return_conditional_losses_3438954o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������:����������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1
�
d
F__inference_dropout_9_layer_call_and_return_conditional_losses_3438763

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
C__inference_FS-CDR_layer_call_and_return_conditional_losses_3438783

inputs
inputs_1&
pairencoder_3438704:	�@!
pairencoder_3438706:@%
pairencoder_3438708:@@!
pairencoder_3438710:@%
pairencoder_3438712:@@!
pairencoder_3438714:@#
dense_12_3438729:	�@
dense_12_3438731:@"
dense_13_3438753:@@
dense_13_3438755:@"
dense_14_3438777:@
dense_14_3438779:
identity�� dense_12/StatefulPartitionedCall� dense_13/StatefulPartitionedCall� dense_14/StatefulPartitionedCall�#pairEncoder/StatefulPartitionedCall�
#pairEncoder/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1pairencoder_3438704pairencoder_3438706pairencoder_3438708pairencoder_3438710pairencoder_3438712pairencoder_3438714*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_pairEncoder_layer_call_and_return_conditional_losses_3438562�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall,pairEncoder/StatefulPartitionedCall:output:0dense_12_3438729dense_12_3438731*
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
GPU 2J 8� *N
fIRG
E__inference_dense_12_layer_call_and_return_conditional_losses_3438728�
dropout_8/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_8_layer_call_and_return_conditional_losses_3438739�
 dense_13/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0dense_13_3438753dense_13_3438755*
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
GPU 2J 8� *N
fIRG
E__inference_dense_13_layer_call_and_return_conditional_losses_3438752�
dropout_9/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_9_layer_call_and_return_conditional_losses_3438763�
 dense_14/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0dense_14_3438777dense_14_3438779*
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
GPU 2J 8� *N
fIRG
E__inference_dense_14_layer_call_and_return_conditional_losses_3438776x
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall$^pairEncoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������:����������: : : : : : : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2J
#pairEncoder/StatefulPartitionedCall#pairEncoder/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
>
input_102
serving_default_input_10:0����������
<
input_91
serving_default_input_9:0����������<
dense_140
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_network
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses"
_tf_keras_layer
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&_random_generator
'__call__
*(&call_and_return_all_conditional_losses"
_tf_keras_layer
�

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses"
_tf_keras_layer
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5_random_generator
6__call__
*7&call_and_return_all_conditional_losses"
_tf_keras_layer
�

8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses"
_tf_keras_layer
�
@iter

Abeta_1

Bbeta_2
	Cdecaym�m�)m�*m�8m�9m�v�v�)v�*v�8v�9v�"
	optimizer
v
D0
E1
F2
G3
H4
I5
6
7
)8
*9
810
911"
trackable_list_wrapper
J
0
1
)2
*3
84
95"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
(__inference_FS-CDR_layer_call_fn_3438810
(__inference_FS-CDR_layer_call_fn_3439115
(__inference_FS-CDR_layer_call_fn_3439145
(__inference_FS-CDR_layer_call_fn_3439011�
���
FullArgSpec1
args)�&
jself
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_FS-CDR_layer_call_and_return_conditional_losses_3439198
C__inference_FS-CDR_layer_call_and_return_conditional_losses_3439279
C__inference_FS-CDR_layer_call_and_return_conditional_losses_3439046
C__inference_FS-CDR_layer_call_and_return_conditional_losses_3439081�
���
FullArgSpec1
args)�&
jself
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
"__inference__wrapped_model_3438252input_9input_10"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
,
Oserving_default"
signature_map
�
Player-0
Qlayer_with_weights-0
Qlayer-1
Rlayer-2
Slayer_with_weights-1
Slayer-3
Tlayer-4
Ulayer_with_weights-2
Ulayer-5
#V_self_saveable_object_factories
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses"
_tf_keras_network
�
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses"
_tf_keras_layer
J
D0
E1
F2
G3
H4
I5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
-__inference_pairEncoder_layer_call_fn_3438577
-__inference_pairEncoder_layer_call_fn_3439329
-__inference_pairEncoder_layer_call_fn_3439347
-__inference_pairEncoder_layer_call_fn_3438657�
���
FullArgSpec1
args)�&
jself
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_pairEncoder_layer_call_and_return_conditional_losses_3439377
H__inference_pairEncoder_layer_call_and_return_conditional_losses_3439421
H__inference_pairEncoder_layer_call_and_return_conditional_losses_3438676
H__inference_pairEncoder_layer_call_and_return_conditional_losses_3438695�
���
FullArgSpec1
args)�&
jself
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
": 	�@2dense_12/kernel
:@2dense_12/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
�2�
*__inference_dense_12_layer_call_fn_3439430�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
E__inference_dense_12_layer_call_and_return_conditional_losses_3439441�
���
FullArgSpec
args�
jself
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
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
"	variables
#trainable_variables
$regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
�2�
+__inference_dropout_8_layer_call_fn_3439446
+__inference_dropout_8_layer_call_fn_3439451�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_dropout_8_layer_call_and_return_conditional_losses_3439456
F__inference_dropout_8_layer_call_and_return_conditional_losses_3439468�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
!:@@2dense_13/kernel
:@2dense_13/bias
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
�2�
*__inference_dense_13_layer_call_fn_3439477�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
E__inference_dense_13_layer_call_and_return_conditional_losses_3439488�
���
FullArgSpec
args�
jself
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
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
1	variables
2trainable_variables
3regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
�2�
+__inference_dropout_9_layer_call_fn_3439493
+__inference_dropout_9_layer_call_fn_3439498�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_dropout_9_layer_call_and_return_conditional_losses_3439503
F__inference_dropout_9_layer_call_and_return_conditional_losses_3439515�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
!:@2dense_14/kernel
:2dense_14/bias
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
�
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
�layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
�2�
*__inference_dense_14_layer_call_fn_3439524�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
E__inference_dense_14_layer_call_and_return_conditional_losses_3439535�
���
FullArgSpec
args�
jself
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
annotations� *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
:	�@2dense/kernel
:@2
dense/bias
 :@@2dense_1/kernel
:@2dense_1/bias
 :@@2dense_2/kernel
:@2dense_2/bias
J
D0
E1
F2
G3
H4
I5"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_signature_wrapper_3439311input_10input_9"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
E
$�_self_saveable_object_factories"
_tf_keras_input_layer
�

Dkernel
Ebias
$�_self_saveable_object_factories
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
$�_self_saveable_object_factories
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�_random_generator
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

Fkernel
Gbias
$�_self_saveable_object_factories
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
$�_self_saveable_object_factories
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�_random_generator
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

Hkernel
Ibias
$�_self_saveable_object_factories
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_dict_wrapper
J
D0
E1
F2
G3
H4
I5"
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
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
�2�
3__inference_cellLineExtractor_layer_call_fn_3438340
3__inference_cellLineExtractor_layer_call_fn_3439552
3__inference_cellLineExtractor_layer_call_fn_3439569
3__inference_cellLineExtractor_layer_call_fn_3438488�
���
FullArgSpec1
args)�&
jself
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_3439596
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_3439637
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_3438509
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_3438530�
���
FullArgSpec1
args)�&
jself
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
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
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
�2�
/__inference_concatenate_4_layer_call_fn_3439643�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
J__inference_concatenate_4_layer_call_and_return_conditional_losses_3439650�
���
FullArgSpec
args�
jself
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
annotations� *
 
J
D0
E1
F2
G3
H4
I5"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
R

�total

�count
�	variables
�	keras_api"
_tf_keras_metric
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
.
D0
E1"
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
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
'__inference_dense_layer_call_fn_3439659�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
B__inference_dense_layer_call_and_return_conditional_losses_3439670�
���
FullArgSpec
args�
jself
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
annotations� *
 
 "
trackable_dict_wrapper
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
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
�2�
)__inference_dropout_layer_call_fn_3439675
)__inference_dropout_layer_call_fn_3439680�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
D__inference_dropout_layer_call_and_return_conditional_losses_3439685
D__inference_dropout_layer_call_and_return_conditional_losses_3439697�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_dict_wrapper
.
F0
G1"
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
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
)__inference_dense_1_layer_call_fn_3439706�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
D__inference_dense_1_layer_call_and_return_conditional_losses_3439717�
���
FullArgSpec
args�
jself
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
annotations� *
 
 "
trackable_dict_wrapper
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
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
�2�
+__inference_dropout_1_layer_call_fn_3439722
+__inference_dropout_1_layer_call_fn_3439727�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_dropout_1_layer_call_and_return_conditional_losses_3439732
F__inference_dropout_1_layer_call_and_return_conditional_losses_3439744�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_dict_wrapper
.
H0
I1"
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
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
)__inference_dense_2_layer_call_fn_3439753�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
D__inference_dense_2_layer_call_and_return_conditional_losses_3439764�
���
FullArgSpec
args�
jself
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
annotations� *
 
J
D0
E1
F2
G3
H4
I5"
trackable_list_wrapper
J
P0
Q1
R2
S3
T4
U5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
':%	�@2Adam/dense_12/kernel/m
 :@2Adam/dense_12/bias/m
&:$@@2Adam/dense_13/kernel/m
 :@2Adam/dense_13/bias/m
&:$@2Adam/dense_14/kernel/m
 :2Adam/dense_14/bias/m
':%	�@2Adam/dense_12/kernel/v
 :@2Adam/dense_12/bias/v
&:$@@2Adam/dense_13/kernel/v
 :@2Adam/dense_13/bias/v
&:$@2Adam/dense_14/kernel/v
 :2Adam/dense_14/bias/v�
C__inference_FS-CDR_layer_call_and_return_conditional_losses_3439046�DEFGHI)*89c�`
Y�V
L�I
"�
input_9����������
#� 
input_10����������
p 

 
� "%�"
�
0���������
� �
C__inference_FS-CDR_layer_call_and_return_conditional_losses_3439081�DEFGHI)*89c�`
Y�V
L�I
"�
input_9����������
#� 
input_10����������
p

 
� "%�"
�
0���������
� �
C__inference_FS-CDR_layer_call_and_return_conditional_losses_3439198�DEFGHI)*89d�a
Z�W
M�J
#� 
inputs/0����������
#� 
inputs/1����������
p 

 
� "%�"
�
0���������
� �
C__inference_FS-CDR_layer_call_and_return_conditional_losses_3439279�DEFGHI)*89d�a
Z�W
M�J
#� 
inputs/0����������
#� 
inputs/1����������
p

 
� "%�"
�
0���������
� �
(__inference_FS-CDR_layer_call_fn_3438810�DEFGHI)*89c�`
Y�V
L�I
"�
input_9����������
#� 
input_10����������
p 

 
� "�����������
(__inference_FS-CDR_layer_call_fn_3439011�DEFGHI)*89c�`
Y�V
L�I
"�
input_9����������
#� 
input_10����������
p

 
� "�����������
(__inference_FS-CDR_layer_call_fn_3439115�DEFGHI)*89d�a
Z�W
M�J
#� 
inputs/0����������
#� 
inputs/1����������
p 

 
� "�����������
(__inference_FS-CDR_layer_call_fn_3439145�DEFGHI)*89d�a
Z�W
M�J
#� 
inputs/0����������
#� 
inputs/1����������
p

 
� "�����������
"__inference__wrapped_model_3438252�DEFGHI)*89[�X
Q�N
L�I
"�
input_9����������
#� 
input_10����������
� "3�0
.
dense_14"�
dense_14����������
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_3438509jDEFGHI9�6
/�,
"�
input_1����������
p 

 
� "%�"
�
0���������@
� �
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_3438530jDEFGHI9�6
/�,
"�
input_1����������
p

 
� "%�"
�
0���������@
� �
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_3439596iDEFGHI8�5
.�+
!�
inputs����������
p 

 
� "%�"
�
0���������@
� �
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_3439637iDEFGHI8�5
.�+
!�
inputs����������
p

 
� "%�"
�
0���������@
� �
3__inference_cellLineExtractor_layer_call_fn_3438340]DEFGHI9�6
/�,
"�
input_1����������
p 

 
� "����������@�
3__inference_cellLineExtractor_layer_call_fn_3438488]DEFGHI9�6
/�,
"�
input_1����������
p

 
� "����������@�
3__inference_cellLineExtractor_layer_call_fn_3439552\DEFGHI8�5
.�+
!�
inputs����������
p 

 
� "����������@�
3__inference_cellLineExtractor_layer_call_fn_3439569\DEFGHI8�5
.�+
!�
inputs����������
p

 
� "����������@�
J__inference_concatenate_4_layer_call_and_return_conditional_losses_3439650�[�X
Q�N
L�I
#� 
inputs/0����������
"�
inputs/1���������@
� "&�#
�
0����������
� �
/__inference_concatenate_4_layer_call_fn_3439643x[�X
Q�N
L�I
#� 
inputs/0����������
"�
inputs/1���������@
� "������������
E__inference_dense_12_layer_call_and_return_conditional_losses_3439441]0�-
&�#
!�
inputs����������
� "%�"
�
0���������@
� ~
*__inference_dense_12_layer_call_fn_3439430P0�-
&�#
!�
inputs����������
� "����������@�
E__inference_dense_13_layer_call_and_return_conditional_losses_3439488\)*/�,
%�"
 �
inputs���������@
� "%�"
�
0���������@
� }
*__inference_dense_13_layer_call_fn_3439477O)*/�,
%�"
 �
inputs���������@
� "����������@�
E__inference_dense_14_layer_call_and_return_conditional_losses_3439535\89/�,
%�"
 �
inputs���������@
� "%�"
�
0���������
� }
*__inference_dense_14_layer_call_fn_3439524O89/�,
%�"
 �
inputs���������@
� "�����������
D__inference_dense_1_layer_call_and_return_conditional_losses_3439717\FG/�,
%�"
 �
inputs���������@
� "%�"
�
0���������@
� |
)__inference_dense_1_layer_call_fn_3439706OFG/�,
%�"
 �
inputs���������@
� "����������@�
D__inference_dense_2_layer_call_and_return_conditional_losses_3439764\HI/�,
%�"
 �
inputs���������@
� "%�"
�
0���������@
� |
)__inference_dense_2_layer_call_fn_3439753OHI/�,
%�"
 �
inputs���������@
� "����������@�
B__inference_dense_layer_call_and_return_conditional_losses_3439670]DE0�-
&�#
!�
inputs����������
� "%�"
�
0���������@
� {
'__inference_dense_layer_call_fn_3439659PDE0�-
&�#
!�
inputs����������
� "����������@�
F__inference_dropout_1_layer_call_and_return_conditional_losses_3439732\3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� �
F__inference_dropout_1_layer_call_and_return_conditional_losses_3439744\3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� ~
+__inference_dropout_1_layer_call_fn_3439722O3�0
)�&
 �
inputs���������@
p 
� "����������@~
+__inference_dropout_1_layer_call_fn_3439727O3�0
)�&
 �
inputs���������@
p
� "����������@�
F__inference_dropout_8_layer_call_and_return_conditional_losses_3439456\3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� �
F__inference_dropout_8_layer_call_and_return_conditional_losses_3439468\3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� ~
+__inference_dropout_8_layer_call_fn_3439446O3�0
)�&
 �
inputs���������@
p 
� "����������@~
+__inference_dropout_8_layer_call_fn_3439451O3�0
)�&
 �
inputs���������@
p
� "����������@�
F__inference_dropout_9_layer_call_and_return_conditional_losses_3439503\3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� �
F__inference_dropout_9_layer_call_and_return_conditional_losses_3439515\3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� ~
+__inference_dropout_9_layer_call_fn_3439493O3�0
)�&
 �
inputs���������@
p 
� "����������@~
+__inference_dropout_9_layer_call_fn_3439498O3�0
)�&
 �
inputs���������@
p
� "����������@�
D__inference_dropout_layer_call_and_return_conditional_losses_3439685\3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� �
D__inference_dropout_layer_call_and_return_conditional_losses_3439697\3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� |
)__inference_dropout_layer_call_fn_3439675O3�0
)�&
 �
inputs���������@
p 
� "����������@|
)__inference_dropout_layer_call_fn_3439680O3�0
)�&
 �
inputs���������@
p
� "����������@�
H__inference_pairEncoder_layer_call_and_return_conditional_losses_3438676�DEFGHIc�`
Y�V
L�I
"�
input_9����������
#� 
input_10����������
p 

 
� "&�#
�
0����������
� �
H__inference_pairEncoder_layer_call_and_return_conditional_losses_3438695�DEFGHIc�`
Y�V
L�I
"�
input_9����������
#� 
input_10����������
p

 
� "&�#
�
0����������
� �
H__inference_pairEncoder_layer_call_and_return_conditional_losses_3439377�DEFGHId�a
Z�W
M�J
#� 
inputs/0����������
#� 
inputs/1����������
p 

 
� "&�#
�
0����������
� �
H__inference_pairEncoder_layer_call_and_return_conditional_losses_3439421�DEFGHId�a
Z�W
M�J
#� 
inputs/0����������
#� 
inputs/1����������
p

 
� "&�#
�
0����������
� �
-__inference_pairEncoder_layer_call_fn_3438577�DEFGHIc�`
Y�V
L�I
"�
input_9����������
#� 
input_10����������
p 

 
� "������������
-__inference_pairEncoder_layer_call_fn_3438657�DEFGHIc�`
Y�V
L�I
"�
input_9����������
#� 
input_10����������
p

 
� "������������
-__inference_pairEncoder_layer_call_fn_3439329�DEFGHId�a
Z�W
M�J
#� 
inputs/0����������
#� 
inputs/1����������
p 

 
� "������������
-__inference_pairEncoder_layer_call_fn_3439347�DEFGHId�a
Z�W
M�J
#� 
inputs/0����������
#� 
inputs/1����������
p

 
� "������������
%__inference_signature_wrapper_3439311�DEFGHI)*89m�j
� 
c�`
/
input_10#� 
input_10����������
-
input_9"�
input_9����������"3�0
.
dense_14"�
dense_14���������