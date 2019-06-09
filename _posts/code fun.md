---
typora-root-url: ..
---

### 一、简单trick+模拟

#### 1.恶龙与勇者

##### note：结构体排序、文件IO

**题目描述**

n条恶龙闯入了王国的领土，为了拯救王国的危机，国王任命你前往冒险者工会雇佣勇者前去讨伐恶龙。每条恶龙的战斗力为ai。每个勇者的战斗力为bi，雇佣他的花费为ci。只有当勇者的战斗力大于等于恶龙的战斗力时，勇者才能击败恶龙。因为勇者都是骄傲的，所以勇者们拒绝互相组队（即每个勇者只能独自一人去讨伐恶龙）。勇者们都具有凝聚空气中魔力的能力，因此可以无限次出战。王国需要金币去灾后重建，需要你计算打败所有恶龙的最小花费。

**输入**

第一行输入一个整数T，表示数据的组数。1≤T≤10。 第一行输入n,m，表示龙的数量和冒险者的数量。0<n,m≤10000。 接下来n行，每行一个数字ai表示龙的战斗力。 接下来m行，每行分别为bi和ci，表示冒险者的战斗力和雇佣他的花费。0<=ai,bi,ci≤100000。

**输出**

如果能击退恶龙们，请输出最小的花费，如果不能，请输出"Kingdom fall"。

**样例输入**

```
2
1 1
6
10 3
3 2
10
14
15
9 10
6 7
```

**样例输出**

```
3
Kingdom fall
```

**代码**

```c++
/* 结构体排序，文件I/O*/

#include<iostream>
#include<algorithm>
using namespace std;

int q[10001];
struct ys{
	int a;
	int b;
}y[10001];

bool cmp(ys s1, ys s2)
{
	return s1.b < s2.b;
}

int main()
{
	int T,n,m;
//	freopen("in.txt", "r", stdin);
	cin>>T;
	while(T--)
	{
		cin>>n>>m;
		for(int i=0; i<n; i++)
			cin >> q[i];
		for(int i=0; i<m; i++)
			cin >> y[i].a >> y[i].b;
		sort(y, y+m, cmp);
		int cost = 0;
		int if_fall = 0;
		for(int i=0; i<n; i++)
		{
			int flag = 0;	
			for(int j=0; j<m; j++)
			{
				if(q[i]<=y[j].a)
				{
					cost += y[j].b;
					flag = 1;
					break;
				}
			}
			if(flag==0)
			{
				if_fall = 1;
				break;
			}
		}
		if(if_fall==1)
		{
			cout << "Kingdom fall" << endl;
		}
		else
		{
			cout << cost << endl;
		}
	}
//	fclose(stdin);
	return 0;
} 
```



### 二、暴力、枚举

#### 1.搭积木

**题目描述**

小明最近喜欢搭数字积木，一共有10块积木，每个积木上有一个数字，0~9。

搭积木规则：

每个积木放到其它两个积木的上面，并且一定比下面的两个积木数字小。

最后搭成4层的金字塔形，必须用完所有的积木。

下面是1种合格的搭法：

![TIM截图20190420101828](/img/TIM截图20190420101828-1557584314848.jpg)

**输入**

无

**输出**

请你计算这样的搭法一共有多少种？

请填表示总数目的数字。 注意：你提交的应该是一个整数，不要填写任何多余的内容或说明性文字。

**代码**



#### 2.



### 三、动态规划

#### 1.最长上升子序列（LIS）

**题目描述**

给定N个数，求这N个数的最长上升子序列的**长度**。

**样例输入**

7

2 5 3 4 1 7 6

**样例输出**

4

**分析**

①$O(n^2)$做法：dp动态规划

状态设计：dp[i]代表以a[i]结尾的LIS的长度 

状态转移：dp[i]=max(dp[i], dp[j]+1) (0<=j< i, a[j]< a[i]) 

边界处理：dp[i]=1 (0<=j< n) 

②$O(nlogn)$做法：贪心+二分

a[i]表示第i个数据。 
dp[i]表示表示长度为i+1的LIS结尾元素的最小值。

 利用贪心的思想，对于一个上升子序列，显然当前最后一个元素越小，越有利于添加新的元素，这样LIS长度自然更长。 
因此，我们只需要维护dp数组，其表示的就是长度为i+1的LIS结尾元素的最小值，保证每一位都是最小值，这样子dp数组的长度就是LIS的长度。



#### 2.最长公共子序列(LCS)

参考：https://www.cnblogs.com/wkfvawl/p/9362287.html

**分析**

状态设计：$c[i,j]$表示：$(x1,x2....xi) $和$ (y1,y2...yj)$ 的最长公共子序列的长度。

状态转移：![2012111100085930](/img/2012111100085930-1557584506280.png)

**代码**

```c++
#include<cstdio>
#include<cstring>
#include<algorithm>
using namespace std;
const int N = 1000;
char a[N],b[N];
int dp[N][N];
int main()
{
    int lena,lenb,i,j;
    while(scanf("%s%s",a,b)!=EOF)
    {
        memset(dp,0,sizeof(dp));
        lena=strlen(a);
        lenb=strlen(b);
        for(i=1;i<=lena;i++)
        {
            for(j=1;j<=lenb;j++)
            {
                if(a[i-1]==b[j-1])
                {
                    dp[i][j]=dp[i-1][j-1]+1;
                }
                else
                {
                    dp[i][j]=max(dp[i-1][j],dp[i][j-1]);
                }
            }
        }
        printf("%d\n",dp[lena][lenb]);
    }
    return 0;
}
```

**路径输出**

```c++
#include<cstdio>
#include<cstring>
#include<algorithm>
using namespace std;
const int N = 1010;
char a[N],b[N];
int dp[N][N];
int flag[N][N];
void Print(int i,int j)
{
    if(i==0||j==0)///递归终止条件
    {
        return ;
    }
    if(!flag[i][j])
    {
        Print(i-1,j-1);
        printf("%c",a[i-1]);
    }
    else if(flag[i][j]==1)
    {
        Print(i-1,j);
    }
    else if(flag[i][j]=-1)
    {
        Print(i,j-1);
    }
}
int main()
{
    int lena,lenb,i,j;
    while(scanf("%s%s",a,b)!=EOF)
    {
        memset(dp,0,sizeof(dp));
        memset(flag,0,sizeof(flag));
        lena=strlen(a);
        lenb=strlen(b);
        for(i=1;i<=lena;i++)
        {
            for(j=1;j<=lenb;j++)
            {
                if(a[i-1]==b[j-1])
                {
                    dp[i][j]=dp[i-1][j-1]+1;
                    flag[i][j]=0;///来自于左上方
                }
                else
                {
                    if(dp[i-1][j]>dp[i][j-1])
                    {
                        dp[i][j]=dp[i-1][j];
                        flag[i][j]=1;///来自于左方
                    }
                    else
                    {
                        dp[i][j]=dp[i][j-1];
                        flag[i][j]=-1;///来自于上方
                    }
                }
            }
        }
        Print(lena,lenb);
    }
    return 0;
}
```



#### 3.背包问题

##### （1）0/1背包

**题目**

有n 种不同的物品，每个物品有两个属性，size 体积，value 价值，现在给一个容量为 w 的背包，问最多可带走多少价值的物品。  

##### （2）完全背包



#### 4.非降子序列的个数

**题目** **http://acm.hdu.edu.cn/showproblem.php?pid=2227**

给定一个长度为n(n <= 100000)的整数序列，求其中的非降子序列的个数。

**分析**

①如果n的值比较小，直接DP求解，时间复杂度$O(n^2)$。设dp[i]表示以a[i]为结尾非降子序列的个数，其状态转移方程为：

![img](img/20140120165241234)

②当n的值比较大，dp+树状数组

**代码**

```c++
#include <iostream>
#include <string.h>
#include <algorithm>
#include <stdio.h>
 
using namespace std;
const int N = 100005;
const int MOD = 1000000007;
 
struct node
{
    int id,val;
};
 
int n;
node a[N];
int aa[N],c[N],t[N];
 
bool cmp(node a,node b)
{
    return a.val < b.val;
}
 
int Lowbit(int x)
{
    return x & (-x);
}
 
void Update(int t,int val)
{
    for(int i=t; i<=n; i+=Lowbit(i))
    {
        c[i] += val;
        c[i] %= MOD;
    }
}
 
int getSum(int x)
{
    int ans = 0;
    for(int i=x; i>0; i-=Lowbit(i))
    {
        ans += c[i];
        ans %= MOD;
    }
    return ans;
}
 
 
int main()
{
    while(scanf("%d",&n)!=EOF)
    {
        memset(c,0,sizeof(c));
        memset(aa,0,sizeof(aa));
        for(int i=1;i<=n;i++)
        {
            scanf("%d",&a[i].val);
            a[i].id = i;
        }
        sort(a+1,a+n+1,cmp);
        aa[a[1].id] = 1;
        for(int i=2;i<=n;i++)
        {
            if(a[i].val != a[i-1].val)
                aa[a[i].id] = i;
            else
                aa[a[i].id] = aa[a[i-1].id];
        }
        for(int i=1;i<=n;i++)
        {
            t[i] = getSum(aa[i]);
            Update(aa[i],t[i]+1);
        }
        printf("%d\n",getSum(n));
    }
    return 0;
}

```



### 四、搜索

1.



### 五、数学基础

#### 1.最大公因数、最小公倍数



#### 2.异或性质

（1）**&** 与

（2）**|** 或

（3）**^** 异或

&nbsp;&nbsp; ①一个数和0异或为本身

&nbsp;&nbsp;&nbsp;②一个数和本身异或为0

应用：求一堆数中独一无二的数







### 六、STL



### 七、各种Trick

1.快速幂



2.素数筛选



### 八、note

1.浮点数不能直接相等，要么转化成整数，要么定义eps。

2.注意什么时候需要用longlong数据类型。



