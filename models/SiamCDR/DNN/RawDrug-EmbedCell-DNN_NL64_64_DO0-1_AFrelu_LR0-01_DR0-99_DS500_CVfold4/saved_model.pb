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
dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@* 
shared_namedense_21/kernel
t
#dense_21/kernel/Read/ReadVariableOpReadVariableOpdense_21/kernel*
_output_shapes
:	�@*
dtype0
r
dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_21/bias
k
!dense_21/bias/Read/ReadVariableOpReadVariableOpdense_21/bias*
_output_shapes
:@*
dtype0
z
dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_22/kernel
s
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel*
_output_shapes

:@@*
dtype0
r
dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_22/bias
k
!dense_22/bias/Read/ReadVariableOpReadVariableOpdense_22/bias*
_output_shapes
:@*
dtype0
z
dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_23/kernel
s
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel*
_output_shapes

:@*
dtype0
r
dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_23/bias
k
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
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
Adam/dense_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*'
shared_nameAdam/dense_21/kernel/m
�
*Adam/dense_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/m*
_output_shapes
:	�@*
dtype0
�
Adam/dense_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_21/bias/m
y
(Adam/dense_21/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/m*
_output_shapes
:@*
dtype0
�
Adam/dense_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_22/kernel/m
�
*Adam/dense_22/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/m*
_output_shapes

:@@*
dtype0
�
Adam/dense_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_22/bias/m
y
(Adam/dense_22/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/m*
_output_shapes
:@*
dtype0
�
Adam/dense_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_23/kernel/m
�
*Adam/dense_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/m*
_output_shapes

:@*
dtype0
�
Adam/dense_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_23/bias/m
y
(Adam/dense_23/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*'
shared_nameAdam/dense_21/kernel/v
�
*Adam/dense_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/v*
_output_shapes
:	�@*
dtype0
�
Adam/dense_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_21/bias/v
y
(Adam/dense_21/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/v*
_output_shapes
:@*
dtype0
�
Adam/dense_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_22/kernel/v
�
*Adam/dense_22/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/v*
_output_shapes

:@@*
dtype0
�
Adam/dense_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_22/bias/v
y
(Adam/dense_22/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/v*
_output_shapes
:@*
dtype0
�
Adam/dense_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_23/kernel/v
�
*Adam/dense_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/v*
_output_shapes

:@*
dtype0
�
Adam/dense_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_23/bias/v
y
(Adam/dense_23/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/v*
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
VARIABLE_VALUEdense_21/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_21/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_22/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_22/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_23/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_23/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEAdam/dense_21/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_21/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_22/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_22/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_23/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_23/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_21/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_21/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_22/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_22/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_23/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_23/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
serving_default_input_15Placeholder*(
_output_shapes
:����������*
dtype0*
shape:����������
}
serving_default_input_16Placeholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_15serving_default_input_16dense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_21/kerneldense_21/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/bias*
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
%__inference_signature_wrapper_6003773
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_21/kernel/Read/ReadVariableOp!dense_21/bias/Read/ReadVariableOp#dense_22/kernel/Read/ReadVariableOp!dense_22/bias/Read/ReadVariableOp#dense_23/kernel/Read/ReadVariableOp!dense_23/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_21/kernel/m/Read/ReadVariableOp(Adam/dense_21/bias/m/Read/ReadVariableOp*Adam/dense_22/kernel/m/Read/ReadVariableOp(Adam/dense_22/bias/m/Read/ReadVariableOp*Adam/dense_23/kernel/m/Read/ReadVariableOp(Adam/dense_23/bias/m/Read/ReadVariableOp*Adam/dense_21/kernel/v/Read/ReadVariableOp(Adam/dense_21/bias/v/Read/ReadVariableOp*Adam/dense_22/kernel/v/Read/ReadVariableOp(Adam/dense_22/bias/v/Read/ReadVariableOp*Adam/dense_23/kernel/v/Read/ReadVariableOp(Adam/dense_23/bias/v/Read/ReadVariableOpConst*+
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
 __inference__traced_save_6004340
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_21/kerneldense_21/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decaydense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biastotalcountAdam/dense_21/kernel/mAdam/dense_21/bias/mAdam/dense_22/kernel/mAdam/dense_22/bias/mAdam/dense_23/kernel/mAdam/dense_23/bias/mAdam/dense_21/kernel/vAdam/dense_21/bias/vAdam/dense_22/kernel/vAdam/dense_22/bias/vAdam/dense_23/kernel/vAdam/dense_23/bias/v**
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
#__inference__traced_restore_6004440��

�w
�
#__inference__traced_restore_6004440
file_prefix3
 assignvariableop_dense_21_kernel:	�@.
 assignvariableop_1_dense_21_bias:@4
"assignvariableop_2_dense_22_kernel:@@.
 assignvariableop_3_dense_22_bias:@4
"assignvariableop_4_dense_23_kernel:@.
 assignvariableop_5_dense_23_bias:&
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
*assignvariableop_18_adam_dense_21_kernel_m:	�@6
(assignvariableop_19_adam_dense_21_bias_m:@<
*assignvariableop_20_adam_dense_22_kernel_m:@@6
(assignvariableop_21_adam_dense_22_bias_m:@<
*assignvariableop_22_adam_dense_23_kernel_m:@6
(assignvariableop_23_adam_dense_23_bias_m:=
*assignvariableop_24_adam_dense_21_kernel_v:	�@6
(assignvariableop_25_adam_dense_21_bias_v:@<
*assignvariableop_26_adam_dense_22_kernel_v:@@6
(assignvariableop_27_adam_dense_22_bias_v:@<
*assignvariableop_28_adam_dense_23_kernel_v:@6
(assignvariableop_29_adam_dense_23_bias_v:
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
AssignVariableOpAssignVariableOp assignvariableop_dense_21_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_21_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_22_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_22_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_23_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_23_biasIdentity_5:output:0"/device:CPU:0*
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
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_dense_21_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_dense_21_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_dense_22_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_dense_22_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_dense_23_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_dense_23_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_dense_21_kernel_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_dense_21_bias_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_dense_22_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_dense_22_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_dense_23_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_dense_23_bias_vIdentity_29:output:0"/device:CPU:0*
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
E__inference_dense_21_layer_call_and_return_conditional_losses_6003190

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
e
F__inference_dropout_1_layer_call_and_return_conditional_losses_6002832

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
�
-__inference_pairEncoder_layer_call_fn_6003791
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
H__inference_pairEncoder_layer_call_and_return_conditional_losses_6003024p
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

�
E__inference_dense_23_layer_call_and_return_conditional_losses_6003238

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
�*
�
H__inference_pairEncoder_layer_call_and_return_conditional_losses_6003839
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
concatenate_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_7/concatConcatV2inputs_0,cellLineExtractor/dense_2/Relu:activations:0"concatenate_7/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������m
IdentityIdentityconcatenate_7/concat:output:0^NoOp*
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
�
e
G__inference_dropout_15_layer_call_and_return_conditional_losses_6003965

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
�
b
)__inference_dropout_layer_call_fn_6004142

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
D__inference_dropout_layer_call_and_return_conditional_losses_6002865o
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
e
F__inference_dropout_1_layer_call_and_return_conditional_losses_6004206

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
t
J__inference_concatenate_7_layer_call_and_return_conditional_losses_6003021

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
�"
�
C__inference_FS-CDR_layer_call_and_return_conditional_losses_6003543
input_15
input_16&
pairencoder_6003512:	�@!
pairencoder_6003514:@%
pairencoder_6003516:@@!
pairencoder_6003518:@%
pairencoder_6003520:@@!
pairencoder_6003522:@#
dense_21_6003525:	�@
dense_21_6003527:@"
dense_22_6003531:@@
dense_22_6003533:@"
dense_23_6003537:@
dense_23_6003539:
identity�� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall�"dropout_14/StatefulPartitionedCall�"dropout_15/StatefulPartitionedCall�#pairEncoder/StatefulPartitionedCall�
#pairEncoder/StatefulPartitionedCallStatefulPartitionedCallinput_15input_16pairencoder_6003512pairencoder_6003514pairencoder_6003516pairencoder_6003518pairencoder_6003520pairencoder_6003522*
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
H__inference_pairEncoder_layer_call_and_return_conditional_losses_6003086�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall,pairEncoder/StatefulPartitionedCall:output:0dense_21_6003525dense_21_6003527*
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
E__inference_dense_21_layer_call_and_return_conditional_losses_6003190�
"dropout_14/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *P
fKRI
G__inference_dropout_14_layer_call_and_return_conditional_losses_6003335�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall+dropout_14/StatefulPartitionedCall:output:0dense_22_6003531dense_22_6003533*
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
E__inference_dense_22_layer_call_and_return_conditional_losses_6003214�
"dropout_15/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0#^dropout_14/StatefulPartitionedCall*
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
GPU 2J 8� *P
fKRI
G__inference_dropout_15_layer_call_and_return_conditional_losses_6003302�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall+dropout_15/StatefulPartitionedCall:output:0dense_23_6003537dense_23_6003539*
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
E__inference_dense_23_layer_call_and_return_conditional_losses_6003238x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall#^dropout_14/StatefulPartitionedCall#^dropout_15/StatefulPartitionedCall$^pairEncoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������:����������: : : : : : : : : : : : 2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2H
"dropout_14/StatefulPartitionedCall"dropout_14/StatefulPartitionedCall2H
"dropout_15/StatefulPartitionedCall"dropout_15/StatefulPartitionedCall2J
#pairEncoder/StatefulPartitionedCall#pairEncoder/StatefulPartitionedCall:R N
(
_output_shapes
:����������
"
_user_specified_name
input_15:RN
(
_output_shapes
:����������
"
_user_specified_name
input_16
�
�
H__inference_pairEncoder_layer_call_and_return_conditional_losses_6003024

inputs
inputs_1,
celllineextractor_6003001:	�@'
celllineextractor_6003003:@+
celllineextractor_6003005:@@'
celllineextractor_6003007:@+
celllineextractor_6003009:@@'
celllineextractor_6003011:@
identity��)cellLineExtractor/StatefulPartitionedCall�
)cellLineExtractor/StatefulPartitionedCallStatefulPartitionedCallinputs_1celllineextractor_6003001celllineextractor_6003003celllineextractor_6003005celllineextractor_6003007celllineextractor_6003009celllineextractor_6003011*
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
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_6002787�
concatenate_7/PartitionedCallPartitionedCallinputs2cellLineExtractor/StatefulPartitionedCall:output:0*
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
J__inference_concatenate_7_layer_call_and_return_conditional_losses_6003021v
IdentityIdentity&concatenate_7/PartitionedCall:output:0^NoOp*
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
�
�
C__inference_FS-CDR_layer_call_and_return_conditional_losses_6003245

inputs
inputs_1&
pairencoder_6003166:	�@!
pairencoder_6003168:@%
pairencoder_6003170:@@!
pairencoder_6003172:@%
pairencoder_6003174:@@!
pairencoder_6003176:@#
dense_21_6003191:	�@
dense_21_6003193:@"
dense_22_6003215:@@
dense_22_6003217:@"
dense_23_6003239:@
dense_23_6003241:
identity�� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall�#pairEncoder/StatefulPartitionedCall�
#pairEncoder/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1pairencoder_6003166pairencoder_6003168pairencoder_6003170pairencoder_6003172pairencoder_6003174pairencoder_6003176*
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
H__inference_pairEncoder_layer_call_and_return_conditional_losses_6003024�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall,pairEncoder/StatefulPartitionedCall:output:0dense_21_6003191dense_21_6003193*
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
E__inference_dense_21_layer_call_and_return_conditional_losses_6003190�
dropout_14/PartitionedCallPartitionedCall)dense_21/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *P
fKRI
G__inference_dropout_14_layer_call_and_return_conditional_losses_6003201�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall#dropout_14/PartitionedCall:output:0dense_22_6003215dense_22_6003217*
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
E__inference_dense_22_layer_call_and_return_conditional_losses_6003214�
dropout_15/PartitionedCallPartitionedCall)dense_22/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *P
fKRI
G__inference_dropout_15_layer_call_and_return_conditional_losses_6003225�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall#dropout_15/PartitionedCall:output:0dense_23_6003239dense_23_6003241*
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
E__inference_dense_23_layer_call_and_return_conditional_losses_6003238x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall$^pairEncoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������:����������: : : : : : : : : : : : 2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2J
#pairEncoder/StatefulPartitionedCall#pairEncoder/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
-__inference_pairEncoder_layer_call_fn_6003119
input_15
input_16
unknown:	�@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_15input_16unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
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
H__inference_pairEncoder_layer_call_and_return_conditional_losses_6003086p
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
input_15:RN
(
_output_shapes
:����������
"
_user_specified_name
input_16
�=
�
H__inference_pairEncoder_layer_call_and_return_conditional_losses_6003883
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
concatenate_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_7/concatConcatV2inputs_0,cellLineExtractor/dense_2/Relu:activations:0"concatenate_7/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������m
IdentityIdentityconcatenate_7/concat:output:0^NoOp*
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
�p
�
C__inference_FS-CDR_layer_call_and_return_conditional_losses_6003741
inputs_0
inputs_1U
Bpairencoder_celllineextractor_dense_matmul_readvariableop_resource:	�@Q
Cpairencoder_celllineextractor_dense_biasadd_readvariableop_resource:@V
Dpairencoder_celllineextractor_dense_1_matmul_readvariableop_resource:@@S
Epairencoder_celllineextractor_dense_1_biasadd_readvariableop_resource:@V
Dpairencoder_celllineextractor_dense_2_matmul_readvariableop_resource:@@S
Epairencoder_celllineextractor_dense_2_biasadd_readvariableop_resource:@:
'dense_21_matmul_readvariableop_resource:	�@6
(dense_21_biasadd_readvariableop_resource:@9
'dense_22_matmul_readvariableop_resource:@@6
(dense_22_biasadd_readvariableop_resource:@9
'dense_23_matmul_readvariableop_resource:@6
(dense_23_biasadd_readvariableop_resource:
identity��dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOp�:pairEncoder/cellLineExtractor/dense/BiasAdd/ReadVariableOp�9pairEncoder/cellLineExtractor/dense/MatMul/ReadVariableOp�<pairEncoder/cellLineExtractor/dense_1/BiasAdd/ReadVariableOp�;pairEncoder/cellLineExtractor/dense_1/MatMul/ReadVariableOp�<pairEncoder/cellLineExtractor/dense_2/BiasAdd/ReadVariableOp�;pairEncoder/cellLineExtractor/dense_2/MatMul/ReadVariableOp�
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
%pairEncoder/concatenate_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
 pairEncoder/concatenate_7/concatConcatV2inputs_08pairEncoder/cellLineExtractor/dense_2/Relu:activations:0.pairEncoder/concatenate_7/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense_21/MatMulMatMul)pairEncoder/concatenate_7/concat:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*'
_output_shapes
:���������@]
dropout_14/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_14/dropout/MulMuldense_21/Relu:activations:0!dropout_14/dropout/Const:output:0*
T0*'
_output_shapes
:���������@c
dropout_14/dropout/ShapeShapedense_21/Relu:activations:0*
T0*
_output_shapes
:�
/dropout_14/dropout/random_uniform/RandomUniformRandomUniform!dropout_14/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0f
!dropout_14/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_14/dropout/GreaterEqualGreaterEqual8dropout_14/dropout/random_uniform/RandomUniform:output:0*dropout_14/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@�
dropout_14/dropout/CastCast#dropout_14/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@�
dropout_14/dropout/Mul_1Muldropout_14/dropout/Mul:z:0dropout_14/dropout/Cast:y:0*
T0*'
_output_shapes
:���������@�
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_22/MatMulMatMuldropout_14/dropout/Mul_1:z:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*'
_output_shapes
:���������@]
dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_15/dropout/MulMuldense_22/Relu:activations:0!dropout_15/dropout/Const:output:0*
T0*'
_output_shapes
:���������@c
dropout_15/dropout/ShapeShapedense_22/Relu:activations:0*
T0*
_output_shapes
:�
/dropout_15/dropout/random_uniform/RandomUniformRandomUniform!dropout_15/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0f
!dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_15/dropout/GreaterEqualGreaterEqual8dropout_15/dropout/random_uniform/RandomUniform:output:0*dropout_15/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@�
dropout_15/dropout/CastCast#dropout_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@�
dropout_15/dropout/Mul_1Muldropout_15/dropout/Mul:z:0dropout_15/dropout/Cast:y:0*
T0*'
_output_shapes
:���������@�
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_23/MatMulMatMuldropout_15/dropout/Mul_1:z:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
dense_23/SigmoidSigmoiddense_23/BiasAdd:output:0*
T0*'
_output_shapes
:���������c
IdentityIdentitydense_23/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp;^pairEncoder/cellLineExtractor/dense/BiasAdd/ReadVariableOp:^pairEncoder/cellLineExtractor/dense/MatMul/ReadVariableOp=^pairEncoder/cellLineExtractor/dense_1/BiasAdd/ReadVariableOp<^pairEncoder/cellLineExtractor/dense_1/MatMul/ReadVariableOp=^pairEncoder/cellLineExtractor/dense_2/BiasAdd/ReadVariableOp<^pairEncoder/cellLineExtractor/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������:����������: : : : : : : : : : : : 2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2x
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

�
D__inference_dense_1_layer_call_and_return_conditional_losses_6004179

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
�
d
+__inference_dropout_1_layer_call_fn_6004189

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
F__inference_dropout_1_layer_call_and_return_conditional_losses_6002832o
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
�,
�
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_6004099

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
�
�
H__inference_pairEncoder_layer_call_and_return_conditional_losses_6003157
input_15
input_16,
celllineextractor_6003142:	�@'
celllineextractor_6003144:@+
celllineextractor_6003146:@@'
celllineextractor_6003148:@+
celllineextractor_6003150:@@'
celllineextractor_6003152:@
identity��)cellLineExtractor/StatefulPartitionedCall�
)cellLineExtractor/StatefulPartitionedCallStatefulPartitionedCallinput_16celllineextractor_6003142celllineextractor_6003144celllineextractor_6003146celllineextractor_6003148celllineextractor_6003150celllineextractor_6003152*
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
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_6002918�
concatenate_7/PartitionedCallPartitionedCallinput_152cellLineExtractor/StatefulPartitionedCall:output:0*
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
J__inference_concatenate_7_layer_call_and_return_conditional_losses_6003021v
IdentityIdentity&concatenate_7/PartitionedCall:output:0^NoOp*
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
)cellLineExtractor/StatefulPartitionedCall)cellLineExtractor/StatefulPartitionedCall:R N
(
_output_shapes
:����������
"
_user_specified_name
input_15:RN
(
_output_shapes
:����������
"
_user_specified_name
input_16
�
�
C__inference_FS-CDR_layer_call_and_return_conditional_losses_6003508
input_15
input_16&
pairencoder_6003477:	�@!
pairencoder_6003479:@%
pairencoder_6003481:@@!
pairencoder_6003483:@%
pairencoder_6003485:@@!
pairencoder_6003487:@#
dense_21_6003490:	�@
dense_21_6003492:@"
dense_22_6003496:@@
dense_22_6003498:@"
dense_23_6003502:@
dense_23_6003504:
identity�� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall�#pairEncoder/StatefulPartitionedCall�
#pairEncoder/StatefulPartitionedCallStatefulPartitionedCallinput_15input_16pairencoder_6003477pairencoder_6003479pairencoder_6003481pairencoder_6003483pairencoder_6003485pairencoder_6003487*
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
H__inference_pairEncoder_layer_call_and_return_conditional_losses_6003024�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall,pairEncoder/StatefulPartitionedCall:output:0dense_21_6003490dense_21_6003492*
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
E__inference_dense_21_layer_call_and_return_conditional_losses_6003190�
dropout_14/PartitionedCallPartitionedCall)dense_21/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *P
fKRI
G__inference_dropout_14_layer_call_and_return_conditional_losses_6003201�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall#dropout_14/PartitionedCall:output:0dense_22_6003496dense_22_6003498*
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
E__inference_dense_22_layer_call_and_return_conditional_losses_6003214�
dropout_15/PartitionedCallPartitionedCall)dense_22/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *P
fKRI
G__inference_dropout_15_layer_call_and_return_conditional_losses_6003225�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall#dropout_15/PartitionedCall:output:0dense_23_6003502dense_23_6003504*
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
E__inference_dense_23_layer_call_and_return_conditional_losses_6003238x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall$^pairEncoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������:����������: : : : : : : : : : : : 2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2J
#pairEncoder/StatefulPartitionedCall#pairEncoder/StatefulPartitionedCall:R N
(
_output_shapes
:����������
"
_user_specified_name
input_15:RN
(
_output_shapes
:����������
"
_user_specified_name
input_16
�
�
%__inference_signature_wrapper_6003773
input_15
input_16
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
StatefulPartitionedCallStatefulPartitionedCallinput_15input_16unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
"__inference__wrapped_model_6002714o
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
input_15:RN
(
_output_shapes
:����������
"
_user_specified_name
input_16
�	
�
3__inference_cellLineExtractor_layer_call_fn_6002950
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
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_6002918o
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
�?
�
 __inference__traced_save_6004340
file_prefix.
*savev2_dense_21_kernel_read_readvariableop,
(savev2_dense_21_bias_read_readvariableop.
*savev2_dense_22_kernel_read_readvariableop,
(savev2_dense_22_bias_read_readvariableop.
*savev2_dense_23_kernel_read_readvariableop,
(savev2_dense_23_bias_read_readvariableop(
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
1savev2_adam_dense_21_kernel_m_read_readvariableop3
/savev2_adam_dense_21_bias_m_read_readvariableop5
1savev2_adam_dense_22_kernel_m_read_readvariableop3
/savev2_adam_dense_22_bias_m_read_readvariableop5
1savev2_adam_dense_23_kernel_m_read_readvariableop3
/savev2_adam_dense_23_bias_m_read_readvariableop5
1savev2_adam_dense_21_kernel_v_read_readvariableop3
/savev2_adam_dense_21_bias_v_read_readvariableop5
1savev2_adam_dense_22_kernel_v_read_readvariableop3
/savev2_adam_dense_22_bias_v_read_readvariableop5
1savev2_adam_dense_23_kernel_v_read_readvariableop3
/savev2_adam_dense_23_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_21_kernel_read_readvariableop(savev2_dense_21_bias_read_readvariableop*savev2_dense_22_kernel_read_readvariableop(savev2_dense_22_bias_read_readvariableop*savev2_dense_23_kernel_read_readvariableop(savev2_dense_23_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_21_kernel_m_read_readvariableop/savev2_adam_dense_21_bias_m_read_readvariableop1savev2_adam_dense_22_kernel_m_read_readvariableop/savev2_adam_dense_22_bias_m_read_readvariableop1savev2_adam_dense_23_kernel_m_read_readvariableop/savev2_adam_dense_23_bias_m_read_readvariableop1savev2_adam_dense_21_kernel_v_read_readvariableop/savev2_adam_dense_21_bias_v_read_readvariableop1savev2_adam_dense_22_kernel_v_read_readvariableop/savev2_adam_dense_22_bias_v_read_readvariableop1savev2_adam_dense_23_kernel_v_read_readvariableop/savev2_adam_dense_23_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
�

�
B__inference_dense_layer_call_and_return_conditional_losses_6002732

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
�
�
)__inference_dense_1_layer_call_fn_6004168

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
D__inference_dense_1_layer_call_and_return_conditional_losses_6002756o
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
�
(__inference_FS-CDR_layer_call_fn_6003272
input_15
input_16
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
StatefulPartitionedCallStatefulPartitionedCallinput_15input_16unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
C__inference_FS-CDR_layer_call_and_return_conditional_losses_6003245o
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
input_15:RN
(
_output_shapes
:����������
"
_user_specified_name
input_16
�
�
*__inference_dense_21_layer_call_fn_6003892

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
E__inference_dense_21_layer_call_and_return_conditional_losses_6003190o
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
�

�
D__inference_dense_2_layer_call_and_return_conditional_losses_6002780

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
c
D__inference_dropout_layer_call_and_return_conditional_losses_6002865

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

�
D__inference_dense_1_layer_call_and_return_conditional_losses_6002756

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
B__inference_dense_layer_call_and_return_conditional_losses_6004132

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
�
e
G__inference_dropout_14_layer_call_and_return_conditional_losses_6003918

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
�
[
/__inference_concatenate_7_layer_call_fn_6004105
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
J__inference_concatenate_7_layer_call_and_return_conditional_losses_6003021a
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
�
�
(__inference_FS-CDR_layer_call_fn_6003607
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
C__inference_FS-CDR_layer_call_and_return_conditional_losses_6003416o
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
e
G__inference_dropout_14_layer_call_and_return_conditional_losses_6003201

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
F__inference_dropout_1_layer_call_and_return_conditional_losses_6002767

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

�
E__inference_dense_22_layer_call_and_return_conditional_losses_6003214

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
�
�
*__inference_dense_23_layer_call_fn_6003986

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
E__inference_dense_23_layer_call_and_return_conditional_losses_6003238o
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
�
H
,__inference_dropout_15_layer_call_fn_6003955

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
GPU 2J 8� *P
fKRI
G__inference_dropout_15_layer_call_and_return_conditional_losses_6003225`
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
�
�
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_6002918

inputs 
dense_6002900:	�@
dense_6002902:@!
dense_1_6002906:@@
dense_1_6002908:@!
dense_2_6002912:@@
dense_2_6002914:@
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dropout/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_6002900dense_6002902*
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
B__inference_dense_layer_call_and_return_conditional_losses_6002732�
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
D__inference_dropout_layer_call_and_return_conditional_losses_6002865�
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_6002906dense_1_6002908*
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
D__inference_dense_1_layer_call_and_return_conditional_losses_6002756�
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
F__inference_dropout_1_layer_call_and_return_conditional_losses_6002832�
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_6002912dense_2_6002914*
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
D__inference_dense_2_layer_call_and_return_conditional_losses_6002780w
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
�
3__inference_cellLineExtractor_layer_call_fn_6004031

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
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_6002918o
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
�
�
'__inference_dense_layer_call_fn_6004121

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
B__inference_dense_layer_call_and_return_conditional_losses_6002732o
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
�
H
,__inference_dropout_14_layer_call_fn_6003908

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
GPU 2J 8� *P
fKRI
G__inference_dropout_14_layer_call_and_return_conditional_losses_6003201`
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
�
�
*__inference_dense_22_layer_call_fn_6003939

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
E__inference_dense_22_layer_call_and_return_conditional_losses_6003214o
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
�
b
D__inference_dropout_layer_call_and_return_conditional_losses_6004147

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
3__inference_cellLineExtractor_layer_call_fn_6002802
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
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_6002787o
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
�

�
E__inference_dense_21_layer_call_and_return_conditional_losses_6003903

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
f
G__inference_dropout_15_layer_call_and_return_conditional_losses_6003302

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
�
3__inference_cellLineExtractor_layer_call_fn_6004014

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
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_6002787o
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
�
e
,__inference_dropout_15_layer_call_fn_6003960

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
GPU 2J 8� *P
fKRI
G__inference_dropout_15_layer_call_and_return_conditional_losses_6003302o
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
�
b
D__inference_dropout_layer_call_and_return_conditional_losses_6002743

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
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_6002787

inputs 
dense_6002733:	�@
dense_6002735:@!
dense_1_6002757:@@
dense_1_6002759:@!
dense_2_6002781:@@
dense_2_6002783:@
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_6002733dense_6002735*
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
B__inference_dense_layer_call_and_return_conditional_losses_6002732�
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
D__inference_dropout_layer_call_and_return_conditional_losses_6002743�
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_6002757dense_1_6002759*
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
D__inference_dense_1_layer_call_and_return_conditional_losses_6002756�
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
F__inference_dropout_1_layer_call_and_return_conditional_losses_6002767�
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_2_6002781dense_2_6002783*
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
D__inference_dense_2_layer_call_and_return_conditional_losses_6002780w
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
�
�
(__inference_FS-CDR_layer_call_fn_6003577
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
C__inference_FS-CDR_layer_call_and_return_conditional_losses_6003245o
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
�"
�
C__inference_FS-CDR_layer_call_and_return_conditional_losses_6003416

inputs
inputs_1&
pairencoder_6003385:	�@!
pairencoder_6003387:@%
pairencoder_6003389:@@!
pairencoder_6003391:@%
pairencoder_6003393:@@!
pairencoder_6003395:@#
dense_21_6003398:	�@
dense_21_6003400:@"
dense_22_6003404:@@
dense_22_6003406:@"
dense_23_6003410:@
dense_23_6003412:
identity�� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall�"dropout_14/StatefulPartitionedCall�"dropout_15/StatefulPartitionedCall�#pairEncoder/StatefulPartitionedCall�
#pairEncoder/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1pairencoder_6003385pairencoder_6003387pairencoder_6003389pairencoder_6003391pairencoder_6003393pairencoder_6003395*
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
H__inference_pairEncoder_layer_call_and_return_conditional_losses_6003086�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall,pairEncoder/StatefulPartitionedCall:output:0dense_21_6003398dense_21_6003400*
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
E__inference_dense_21_layer_call_and_return_conditional_losses_6003190�
"dropout_14/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *P
fKRI
G__inference_dropout_14_layer_call_and_return_conditional_losses_6003335�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall+dropout_14/StatefulPartitionedCall:output:0dense_22_6003404dense_22_6003406*
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
E__inference_dense_22_layer_call_and_return_conditional_losses_6003214�
"dropout_15/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0#^dropout_14/StatefulPartitionedCall*
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
GPU 2J 8� *P
fKRI
G__inference_dropout_15_layer_call_and_return_conditional_losses_6003302�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall+dropout_15/StatefulPartitionedCall:output:0dense_23_6003410dense_23_6003412*
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
E__inference_dense_23_layer_call_and_return_conditional_losses_6003238x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall#^dropout_14/StatefulPartitionedCall#^dropout_15/StatefulPartitionedCall$^pairEncoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������:����������: : : : : : : : : : : : 2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2H
"dropout_14/StatefulPartitionedCall"dropout_14/StatefulPartitionedCall2H
"dropout_15/StatefulPartitionedCall"dropout_15/StatefulPartitionedCall2J
#pairEncoder/StatefulPartitionedCall#pairEncoder/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
-__inference_pairEncoder_layer_call_fn_6003809
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
H__inference_pairEncoder_layer_call_and_return_conditional_losses_6003086p
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
�
(__inference_FS-CDR_layer_call_fn_6003473
input_15
input_16
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
StatefulPartitionedCallStatefulPartitionedCallinput_15input_16unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
C__inference_FS-CDR_layer_call_and_return_conditional_losses_6003416o
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
input_15:RN
(
_output_shapes
:����������
"
_user_specified_name
input_16
�
�
)__inference_dense_2_layer_call_fn_6004215

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
D__inference_dense_2_layer_call_and_return_conditional_losses_6002780o
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
f
G__inference_dropout_15_layer_call_and_return_conditional_losses_6003977

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
f
G__inference_dropout_14_layer_call_and_return_conditional_losses_6003930

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

�
E__inference_dense_23_layer_call_and_return_conditional_losses_6003997

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
�
�
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_6002971
input_1 
dense_6002953:	�@
dense_6002955:@!
dense_1_6002959:@@
dense_1_6002961:@!
dense_2_6002965:@@
dense_2_6002967:@
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_6002953dense_6002955*
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
B__inference_dense_layer_call_and_return_conditional_losses_6002732�
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
D__inference_dropout_layer_call_and_return_conditional_losses_6002743�
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_6002959dense_1_6002961*
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
D__inference_dense_1_layer_call_and_return_conditional_losses_6002756�
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
F__inference_dropout_1_layer_call_and_return_conditional_losses_6002767�
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_2_6002965dense_2_6002967*
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
D__inference_dense_2_layer_call_and_return_conditional_losses_6002780w
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
�
e
,__inference_dropout_14_layer_call_fn_6003913

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
GPU 2J 8� *P
fKRI
G__inference_dropout_14_layer_call_and_return_conditional_losses_6003335o
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
�
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_6004058

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
�
G
+__inference_dropout_1_layer_call_fn_6004184

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
F__inference_dropout_1_layer_call_and_return_conditional_losses_6002767`
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
�
e
G__inference_dropout_15_layer_call_and_return_conditional_losses_6003225

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
f
G__inference_dropout_14_layer_call_and_return_conditional_losses_6003335

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
�
-__inference_pairEncoder_layer_call_fn_6003039
input_15
input_16
unknown:	�@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_15input_16unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
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
H__inference_pairEncoder_layer_call_and_return_conditional_losses_6003024p
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
input_15:RN
(
_output_shapes
:����������
"
_user_specified_name
input_16
�	
c
D__inference_dropout_layer_call_and_return_conditional_losses_6004159

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
"__inference__wrapped_model_6002714
input_15
input_16\
Ifs_cdr_pairencoder_celllineextractor_dense_matmul_readvariableop_resource:	�@X
Jfs_cdr_pairencoder_celllineextractor_dense_biasadd_readvariableop_resource:@]
Kfs_cdr_pairencoder_celllineextractor_dense_1_matmul_readvariableop_resource:@@Z
Lfs_cdr_pairencoder_celllineextractor_dense_1_biasadd_readvariableop_resource:@]
Kfs_cdr_pairencoder_celllineextractor_dense_2_matmul_readvariableop_resource:@@Z
Lfs_cdr_pairencoder_celllineextractor_dense_2_biasadd_readvariableop_resource:@A
.fs_cdr_dense_21_matmul_readvariableop_resource:	�@=
/fs_cdr_dense_21_biasadd_readvariableop_resource:@@
.fs_cdr_dense_22_matmul_readvariableop_resource:@@=
/fs_cdr_dense_22_biasadd_readvariableop_resource:@@
.fs_cdr_dense_23_matmul_readvariableop_resource:@=
/fs_cdr_dense_23_biasadd_readvariableop_resource:
identity��&FS-CDR/dense_21/BiasAdd/ReadVariableOp�%FS-CDR/dense_21/MatMul/ReadVariableOp�&FS-CDR/dense_22/BiasAdd/ReadVariableOp�%FS-CDR/dense_22/MatMul/ReadVariableOp�&FS-CDR/dense_23/BiasAdd/ReadVariableOp�%FS-CDR/dense_23/MatMul/ReadVariableOp�AFS-CDR/pairEncoder/cellLineExtractor/dense/BiasAdd/ReadVariableOp�@FS-CDR/pairEncoder/cellLineExtractor/dense/MatMul/ReadVariableOp�CFS-CDR/pairEncoder/cellLineExtractor/dense_1/BiasAdd/ReadVariableOp�BFS-CDR/pairEncoder/cellLineExtractor/dense_1/MatMul/ReadVariableOp�CFS-CDR/pairEncoder/cellLineExtractor/dense_2/BiasAdd/ReadVariableOp�BFS-CDR/pairEncoder/cellLineExtractor/dense_2/MatMul/ReadVariableOp�
@FS-CDR/pairEncoder/cellLineExtractor/dense/MatMul/ReadVariableOpReadVariableOpIfs_cdr_pairencoder_celllineextractor_dense_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
1FS-CDR/pairEncoder/cellLineExtractor/dense/MatMulMatMulinput_16HFS-CDR/pairEncoder/cellLineExtractor/dense/MatMul/ReadVariableOp:value:0*
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
,FS-CDR/pairEncoder/concatenate_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
'FS-CDR/pairEncoder/concatenate_7/concatConcatV2input_15?FS-CDR/pairEncoder/cellLineExtractor/dense_2/Relu:activations:05FS-CDR/pairEncoder/concatenate_7/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
%FS-CDR/dense_21/MatMul/ReadVariableOpReadVariableOp.fs_cdr_dense_21_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
FS-CDR/dense_21/MatMulMatMul0FS-CDR/pairEncoder/concatenate_7/concat:output:0-FS-CDR/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
&FS-CDR/dense_21/BiasAdd/ReadVariableOpReadVariableOp/fs_cdr_dense_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
FS-CDR/dense_21/BiasAddBiasAdd FS-CDR/dense_21/MatMul:product:0.FS-CDR/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@p
FS-CDR/dense_21/ReluRelu FS-CDR/dense_21/BiasAdd:output:0*
T0*'
_output_shapes
:���������@|
FS-CDR/dropout_14/IdentityIdentity"FS-CDR/dense_21/Relu:activations:0*
T0*'
_output_shapes
:���������@�
%FS-CDR/dense_22/MatMul/ReadVariableOpReadVariableOp.fs_cdr_dense_22_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
FS-CDR/dense_22/MatMulMatMul#FS-CDR/dropout_14/Identity:output:0-FS-CDR/dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
&FS-CDR/dense_22/BiasAdd/ReadVariableOpReadVariableOp/fs_cdr_dense_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
FS-CDR/dense_22/BiasAddBiasAdd FS-CDR/dense_22/MatMul:product:0.FS-CDR/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@p
FS-CDR/dense_22/ReluRelu FS-CDR/dense_22/BiasAdd:output:0*
T0*'
_output_shapes
:���������@|
FS-CDR/dropout_15/IdentityIdentity"FS-CDR/dense_22/Relu:activations:0*
T0*'
_output_shapes
:���������@�
%FS-CDR/dense_23/MatMul/ReadVariableOpReadVariableOp.fs_cdr_dense_23_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
FS-CDR/dense_23/MatMulMatMul#FS-CDR/dropout_15/Identity:output:0-FS-CDR/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
&FS-CDR/dense_23/BiasAdd/ReadVariableOpReadVariableOp/fs_cdr_dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
FS-CDR/dense_23/BiasAddBiasAdd FS-CDR/dense_23/MatMul:product:0.FS-CDR/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
FS-CDR/dense_23/SigmoidSigmoid FS-CDR/dense_23/BiasAdd:output:0*
T0*'
_output_shapes
:���������j
IdentityIdentityFS-CDR/dense_23/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp'^FS-CDR/dense_21/BiasAdd/ReadVariableOp&^FS-CDR/dense_21/MatMul/ReadVariableOp'^FS-CDR/dense_22/BiasAdd/ReadVariableOp&^FS-CDR/dense_22/MatMul/ReadVariableOp'^FS-CDR/dense_23/BiasAdd/ReadVariableOp&^FS-CDR/dense_23/MatMul/ReadVariableOpB^FS-CDR/pairEncoder/cellLineExtractor/dense/BiasAdd/ReadVariableOpA^FS-CDR/pairEncoder/cellLineExtractor/dense/MatMul/ReadVariableOpD^FS-CDR/pairEncoder/cellLineExtractor/dense_1/BiasAdd/ReadVariableOpC^FS-CDR/pairEncoder/cellLineExtractor/dense_1/MatMul/ReadVariableOpD^FS-CDR/pairEncoder/cellLineExtractor/dense_2/BiasAdd/ReadVariableOpC^FS-CDR/pairEncoder/cellLineExtractor/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������:����������: : : : : : : : : : : : 2P
&FS-CDR/dense_21/BiasAdd/ReadVariableOp&FS-CDR/dense_21/BiasAdd/ReadVariableOp2N
%FS-CDR/dense_21/MatMul/ReadVariableOp%FS-CDR/dense_21/MatMul/ReadVariableOp2P
&FS-CDR/dense_22/BiasAdd/ReadVariableOp&FS-CDR/dense_22/BiasAdd/ReadVariableOp2N
%FS-CDR/dense_22/MatMul/ReadVariableOp%FS-CDR/dense_22/MatMul/ReadVariableOp2P
&FS-CDR/dense_23/BiasAdd/ReadVariableOp&FS-CDR/dense_23/BiasAdd/ReadVariableOp2N
%FS-CDR/dense_23/MatMul/ReadVariableOp%FS-CDR/dense_23/MatMul/ReadVariableOp2�
AFS-CDR/pairEncoder/cellLineExtractor/dense/BiasAdd/ReadVariableOpAFS-CDR/pairEncoder/cellLineExtractor/dense/BiasAdd/ReadVariableOp2�
@FS-CDR/pairEncoder/cellLineExtractor/dense/MatMul/ReadVariableOp@FS-CDR/pairEncoder/cellLineExtractor/dense/MatMul/ReadVariableOp2�
CFS-CDR/pairEncoder/cellLineExtractor/dense_1/BiasAdd/ReadVariableOpCFS-CDR/pairEncoder/cellLineExtractor/dense_1/BiasAdd/ReadVariableOp2�
BFS-CDR/pairEncoder/cellLineExtractor/dense_1/MatMul/ReadVariableOpBFS-CDR/pairEncoder/cellLineExtractor/dense_1/MatMul/ReadVariableOp2�
CFS-CDR/pairEncoder/cellLineExtractor/dense_2/BiasAdd/ReadVariableOpCFS-CDR/pairEncoder/cellLineExtractor/dense_2/BiasAdd/ReadVariableOp2�
BFS-CDR/pairEncoder/cellLineExtractor/dense_2/MatMul/ReadVariableOpBFS-CDR/pairEncoder/cellLineExtractor/dense_2/MatMul/ReadVariableOp:R N
(
_output_shapes
:����������
"
_user_specified_name
input_15:RN
(
_output_shapes
:����������
"
_user_specified_name
input_16
�
�
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_6002992
input_1 
dense_6002974:	�@
dense_6002976:@!
dense_1_6002980:@@
dense_1_6002982:@!
dense_2_6002986:@@
dense_2_6002988:@
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dropout/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_6002974dense_6002976*
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
B__inference_dense_layer_call_and_return_conditional_losses_6002732�
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
D__inference_dropout_layer_call_and_return_conditional_losses_6002865�
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_6002980dense_1_6002982*
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
D__inference_dense_1_layer_call_and_return_conditional_losses_6002756�
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
F__inference_dropout_1_layer_call_and_return_conditional_losses_6002832�
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_6002986dense_2_6002988*
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
D__inference_dense_2_layer_call_and_return_conditional_losses_6002780w
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
�
�
H__inference_pairEncoder_layer_call_and_return_conditional_losses_6003138
input_15
input_16,
celllineextractor_6003123:	�@'
celllineextractor_6003125:@+
celllineextractor_6003127:@@'
celllineextractor_6003129:@+
celllineextractor_6003131:@@'
celllineextractor_6003133:@
identity��)cellLineExtractor/StatefulPartitionedCall�
)cellLineExtractor/StatefulPartitionedCallStatefulPartitionedCallinput_16celllineextractor_6003123celllineextractor_6003125celllineextractor_6003127celllineextractor_6003129celllineextractor_6003131celllineextractor_6003133*
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
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_6002787�
concatenate_7/PartitionedCallPartitionedCallinput_152cellLineExtractor/StatefulPartitionedCall:output:0*
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
J__inference_concatenate_7_layer_call_and_return_conditional_losses_6003021v
IdentityIdentity&concatenate_7/PartitionedCall:output:0^NoOp*
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
)cellLineExtractor/StatefulPartitionedCall)cellLineExtractor/StatefulPartitionedCall:R N
(
_output_shapes
:����������
"
_user_specified_name
input_15:RN
(
_output_shapes
:����������
"
_user_specified_name
input_16
�
�
H__inference_pairEncoder_layer_call_and_return_conditional_losses_6003086

inputs
inputs_1,
celllineextractor_6003071:	�@'
celllineextractor_6003073:@+
celllineextractor_6003075:@@'
celllineextractor_6003077:@+
celllineextractor_6003079:@@'
celllineextractor_6003081:@
identity��)cellLineExtractor/StatefulPartitionedCall�
)cellLineExtractor/StatefulPartitionedCallStatefulPartitionedCallinputs_1celllineextractor_6003071celllineextractor_6003073celllineextractor_6003075celllineextractor_6003077celllineextractor_6003079celllineextractor_6003081*
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
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_6002918�
concatenate_7/PartitionedCallPartitionedCallinputs2cellLineExtractor/StatefulPartitionedCall:output:0*
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
J__inference_concatenate_7_layer_call_and_return_conditional_losses_6003021v
IdentityIdentity&concatenate_7/PartitionedCall:output:0^NoOp*
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
�
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_6004194

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

�
D__inference_dense_2_layer_call_and_return_conditional_losses_6004226

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
�K
�
C__inference_FS-CDR_layer_call_and_return_conditional_losses_6003660
inputs_0
inputs_1U
Bpairencoder_celllineextractor_dense_matmul_readvariableop_resource:	�@Q
Cpairencoder_celllineextractor_dense_biasadd_readvariableop_resource:@V
Dpairencoder_celllineextractor_dense_1_matmul_readvariableop_resource:@@S
Epairencoder_celllineextractor_dense_1_biasadd_readvariableop_resource:@V
Dpairencoder_celllineextractor_dense_2_matmul_readvariableop_resource:@@S
Epairencoder_celllineextractor_dense_2_biasadd_readvariableop_resource:@:
'dense_21_matmul_readvariableop_resource:	�@6
(dense_21_biasadd_readvariableop_resource:@9
'dense_22_matmul_readvariableop_resource:@@6
(dense_22_biasadd_readvariableop_resource:@9
'dense_23_matmul_readvariableop_resource:@6
(dense_23_biasadd_readvariableop_resource:
identity��dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOp�:pairEncoder/cellLineExtractor/dense/BiasAdd/ReadVariableOp�9pairEncoder/cellLineExtractor/dense/MatMul/ReadVariableOp�<pairEncoder/cellLineExtractor/dense_1/BiasAdd/ReadVariableOp�;pairEncoder/cellLineExtractor/dense_1/MatMul/ReadVariableOp�<pairEncoder/cellLineExtractor/dense_2/BiasAdd/ReadVariableOp�;pairEncoder/cellLineExtractor/dense_2/MatMul/ReadVariableOp�
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
%pairEncoder/concatenate_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
 pairEncoder/concatenate_7/concatConcatV2inputs_08pairEncoder/cellLineExtractor/dense_2/Relu:activations:0.pairEncoder/concatenate_7/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense_21/MatMulMatMul)pairEncoder/concatenate_7/concat:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*'
_output_shapes
:���������@n
dropout_14/IdentityIdentitydense_21/Relu:activations:0*
T0*'
_output_shapes
:���������@�
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_22/MatMulMatMuldropout_14/Identity:output:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*'
_output_shapes
:���������@n
dropout_15/IdentityIdentitydense_22/Relu:activations:0*
T0*'
_output_shapes
:���������@�
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_23/MatMulMatMuldropout_15/Identity:output:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
dense_23/SigmoidSigmoiddense_23/BiasAdd:output:0*
T0*'
_output_shapes
:���������c
IdentityIdentitydense_23/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp;^pairEncoder/cellLineExtractor/dense/BiasAdd/ReadVariableOp:^pairEncoder/cellLineExtractor/dense/MatMul/ReadVariableOp=^pairEncoder/cellLineExtractor/dense_1/BiasAdd/ReadVariableOp<^pairEncoder/cellLineExtractor/dense_1/MatMul/ReadVariableOp=^pairEncoder/cellLineExtractor/dense_2/BiasAdd/ReadVariableOp<^pairEncoder/cellLineExtractor/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������:����������: : : : : : : : : : : : 2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2x
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

�
E__inference_dense_22_layer_call_and_return_conditional_losses_6003950

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
�
v
J__inference_concatenate_7_layer_call_and_return_conditional_losses_6004112
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
�
E
)__inference_dropout_layer_call_fn_6004137

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
D__inference_dropout_layer_call_and_return_conditional_losses_6002743`
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
input_152
serving_default_input_15:0����������
>
input_162
serving_default_input_16:0����������<
dense_230
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
(__inference_FS-CDR_layer_call_fn_6003272
(__inference_FS-CDR_layer_call_fn_6003577
(__inference_FS-CDR_layer_call_fn_6003607
(__inference_FS-CDR_layer_call_fn_6003473�
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
C__inference_FS-CDR_layer_call_and_return_conditional_losses_6003660
C__inference_FS-CDR_layer_call_and_return_conditional_losses_6003741
C__inference_FS-CDR_layer_call_and_return_conditional_losses_6003508
C__inference_FS-CDR_layer_call_and_return_conditional_losses_6003543�
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
"__inference__wrapped_model_6002714input_15input_16"�
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
-__inference_pairEncoder_layer_call_fn_6003039
-__inference_pairEncoder_layer_call_fn_6003791
-__inference_pairEncoder_layer_call_fn_6003809
-__inference_pairEncoder_layer_call_fn_6003119�
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
H__inference_pairEncoder_layer_call_and_return_conditional_losses_6003839
H__inference_pairEncoder_layer_call_and_return_conditional_losses_6003883
H__inference_pairEncoder_layer_call_and_return_conditional_losses_6003138
H__inference_pairEncoder_layer_call_and_return_conditional_losses_6003157�
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
": 	�@2dense_21/kernel
:@2dense_21/bias
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
*__inference_dense_21_layer_call_fn_6003892�
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
E__inference_dense_21_layer_call_and_return_conditional_losses_6003903�
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
,__inference_dropout_14_layer_call_fn_6003908
,__inference_dropout_14_layer_call_fn_6003913�
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
G__inference_dropout_14_layer_call_and_return_conditional_losses_6003918
G__inference_dropout_14_layer_call_and_return_conditional_losses_6003930�
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
!:@@2dense_22/kernel
:@2dense_22/bias
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
*__inference_dense_22_layer_call_fn_6003939�
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
E__inference_dense_22_layer_call_and_return_conditional_losses_6003950�
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
,__inference_dropout_15_layer_call_fn_6003955
,__inference_dropout_15_layer_call_fn_6003960�
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
G__inference_dropout_15_layer_call_and_return_conditional_losses_6003965
G__inference_dropout_15_layer_call_and_return_conditional_losses_6003977�
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
!:@2dense_23/kernel
:2dense_23/bias
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
*__inference_dense_23_layer_call_fn_6003986�
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
E__inference_dense_23_layer_call_and_return_conditional_losses_6003997�
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
%__inference_signature_wrapper_6003773input_15input_16"�
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
3__inference_cellLineExtractor_layer_call_fn_6002802
3__inference_cellLineExtractor_layer_call_fn_6004014
3__inference_cellLineExtractor_layer_call_fn_6004031
3__inference_cellLineExtractor_layer_call_fn_6002950�
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
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_6004058
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_6004099
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_6002971
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_6002992�
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
/__inference_concatenate_7_layer_call_fn_6004105�
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
J__inference_concatenate_7_layer_call_and_return_conditional_losses_6004112�
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
'__inference_dense_layer_call_fn_6004121�
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
B__inference_dense_layer_call_and_return_conditional_losses_6004132�
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
)__inference_dropout_layer_call_fn_6004137
)__inference_dropout_layer_call_fn_6004142�
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
D__inference_dropout_layer_call_and_return_conditional_losses_6004147
D__inference_dropout_layer_call_and_return_conditional_losses_6004159�
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
)__inference_dense_1_layer_call_fn_6004168�
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
D__inference_dense_1_layer_call_and_return_conditional_losses_6004179�
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
+__inference_dropout_1_layer_call_fn_6004184
+__inference_dropout_1_layer_call_fn_6004189�
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
F__inference_dropout_1_layer_call_and_return_conditional_losses_6004194
F__inference_dropout_1_layer_call_and_return_conditional_losses_6004206�
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
)__inference_dense_2_layer_call_fn_6004215�
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
D__inference_dense_2_layer_call_and_return_conditional_losses_6004226�
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
':%	�@2Adam/dense_21/kernel/m
 :@2Adam/dense_21/bias/m
&:$@@2Adam/dense_22/kernel/m
 :@2Adam/dense_22/bias/m
&:$@2Adam/dense_23/kernel/m
 :2Adam/dense_23/bias/m
':%	�@2Adam/dense_21/kernel/v
 :@2Adam/dense_21/bias/v
&:$@@2Adam/dense_22/kernel/v
 :@2Adam/dense_22/bias/v
&:$@2Adam/dense_23/kernel/v
 :2Adam/dense_23/bias/v�
C__inference_FS-CDR_layer_call_and_return_conditional_losses_6003508�DEFGHI)*89d�a
Z�W
M�J
#� 
input_15����������
#� 
input_16����������
p 

 
� "%�"
�
0���������
� �
C__inference_FS-CDR_layer_call_and_return_conditional_losses_6003543�DEFGHI)*89d�a
Z�W
M�J
#� 
input_15����������
#� 
input_16����������
p

 
� "%�"
�
0���������
� �
C__inference_FS-CDR_layer_call_and_return_conditional_losses_6003660�DEFGHI)*89d�a
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
C__inference_FS-CDR_layer_call_and_return_conditional_losses_6003741�DEFGHI)*89d�a
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
(__inference_FS-CDR_layer_call_fn_6003272�DEFGHI)*89d�a
Z�W
M�J
#� 
input_15����������
#� 
input_16����������
p 

 
� "�����������
(__inference_FS-CDR_layer_call_fn_6003473�DEFGHI)*89d�a
Z�W
M�J
#� 
input_15����������
#� 
input_16����������
p

 
� "�����������
(__inference_FS-CDR_layer_call_fn_6003577�DEFGHI)*89d�a
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
(__inference_FS-CDR_layer_call_fn_6003607�DEFGHI)*89d�a
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
"__inference__wrapped_model_6002714�DEFGHI)*89\�Y
R�O
M�J
#� 
input_15����������
#� 
input_16����������
� "3�0
.
dense_23"�
dense_23����������
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_6002971jDEFGHI9�6
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
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_6002992jDEFGHI9�6
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
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_6004058iDEFGHI8�5
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
N__inference_cellLineExtractor_layer_call_and_return_conditional_losses_6004099iDEFGHI8�5
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
3__inference_cellLineExtractor_layer_call_fn_6002802]DEFGHI9�6
/�,
"�
input_1����������
p 

 
� "����������@�
3__inference_cellLineExtractor_layer_call_fn_6002950]DEFGHI9�6
/�,
"�
input_1����������
p

 
� "����������@�
3__inference_cellLineExtractor_layer_call_fn_6004014\DEFGHI8�5
.�+
!�
inputs����������
p 

 
� "����������@�
3__inference_cellLineExtractor_layer_call_fn_6004031\DEFGHI8�5
.�+
!�
inputs����������
p

 
� "����������@�
J__inference_concatenate_7_layer_call_and_return_conditional_losses_6004112�[�X
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
/__inference_concatenate_7_layer_call_fn_6004105x[�X
Q�N
L�I
#� 
inputs/0����������
"�
inputs/1���������@
� "������������
D__inference_dense_1_layer_call_and_return_conditional_losses_6004179\FG/�,
%�"
 �
inputs���������@
� "%�"
�
0���������@
� |
)__inference_dense_1_layer_call_fn_6004168OFG/�,
%�"
 �
inputs���������@
� "����������@�
E__inference_dense_21_layer_call_and_return_conditional_losses_6003903]0�-
&�#
!�
inputs����������
� "%�"
�
0���������@
� ~
*__inference_dense_21_layer_call_fn_6003892P0�-
&�#
!�
inputs����������
� "����������@�
E__inference_dense_22_layer_call_and_return_conditional_losses_6003950\)*/�,
%�"
 �
inputs���������@
� "%�"
�
0���������@
� }
*__inference_dense_22_layer_call_fn_6003939O)*/�,
%�"
 �
inputs���������@
� "����������@�
E__inference_dense_23_layer_call_and_return_conditional_losses_6003997\89/�,
%�"
 �
inputs���������@
� "%�"
�
0���������
� }
*__inference_dense_23_layer_call_fn_6003986O89/�,
%�"
 �
inputs���������@
� "�����������
D__inference_dense_2_layer_call_and_return_conditional_losses_6004226\HI/�,
%�"
 �
inputs���������@
� "%�"
�
0���������@
� |
)__inference_dense_2_layer_call_fn_6004215OHI/�,
%�"
 �
inputs���������@
� "����������@�
B__inference_dense_layer_call_and_return_conditional_losses_6004132]DE0�-
&�#
!�
inputs����������
� "%�"
�
0���������@
� {
'__inference_dense_layer_call_fn_6004121PDE0�-
&�#
!�
inputs����������
� "����������@�
G__inference_dropout_14_layer_call_and_return_conditional_losses_6003918\3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� �
G__inference_dropout_14_layer_call_and_return_conditional_losses_6003930\3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� 
,__inference_dropout_14_layer_call_fn_6003908O3�0
)�&
 �
inputs���������@
p 
� "����������@
,__inference_dropout_14_layer_call_fn_6003913O3�0
)�&
 �
inputs���������@
p
� "����������@�
G__inference_dropout_15_layer_call_and_return_conditional_losses_6003965\3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� �
G__inference_dropout_15_layer_call_and_return_conditional_losses_6003977\3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� 
,__inference_dropout_15_layer_call_fn_6003955O3�0
)�&
 �
inputs���������@
p 
� "����������@
,__inference_dropout_15_layer_call_fn_6003960O3�0
)�&
 �
inputs���������@
p
� "����������@�
F__inference_dropout_1_layer_call_and_return_conditional_losses_6004194\3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� �
F__inference_dropout_1_layer_call_and_return_conditional_losses_6004206\3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� ~
+__inference_dropout_1_layer_call_fn_6004184O3�0
)�&
 �
inputs���������@
p 
� "����������@~
+__inference_dropout_1_layer_call_fn_6004189O3�0
)�&
 �
inputs���������@
p
� "����������@�
D__inference_dropout_layer_call_and_return_conditional_losses_6004147\3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� �
D__inference_dropout_layer_call_and_return_conditional_losses_6004159\3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� |
)__inference_dropout_layer_call_fn_6004137O3�0
)�&
 �
inputs���������@
p 
� "����������@|
)__inference_dropout_layer_call_fn_6004142O3�0
)�&
 �
inputs���������@
p
� "����������@�
H__inference_pairEncoder_layer_call_and_return_conditional_losses_6003138�DEFGHId�a
Z�W
M�J
#� 
input_15����������
#� 
input_16����������
p 

 
� "&�#
�
0����������
� �
H__inference_pairEncoder_layer_call_and_return_conditional_losses_6003157�DEFGHId�a
Z�W
M�J
#� 
input_15����������
#� 
input_16����������
p

 
� "&�#
�
0����������
� �
H__inference_pairEncoder_layer_call_and_return_conditional_losses_6003839�DEFGHId�a
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
H__inference_pairEncoder_layer_call_and_return_conditional_losses_6003883�DEFGHId�a
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
-__inference_pairEncoder_layer_call_fn_6003039�DEFGHId�a
Z�W
M�J
#� 
input_15����������
#� 
input_16����������
p 

 
� "������������
-__inference_pairEncoder_layer_call_fn_6003119�DEFGHId�a
Z�W
M�J
#� 
input_15����������
#� 
input_16����������
p

 
� "������������
-__inference_pairEncoder_layer_call_fn_6003791�DEFGHId�a
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
-__inference_pairEncoder_layer_call_fn_6003809�DEFGHId�a
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
%__inference_signature_wrapper_6003773�DEFGHI)*89o�l
� 
e�b
/
input_15#� 
input_15����������
/
input_16#� 
input_16����������"3�0
.
dense_23"�
dense_23���������