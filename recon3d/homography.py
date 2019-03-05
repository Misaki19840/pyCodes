#!/usr/bin/python
# -*- coding: utf-8 -*-

from numpy import *

def normalize(points): 
  """ 同次座標系の点の集合を、最後のrow=1になるように正規化する。"""

  for row in points:
    row /= points[-1]
  return points

def make_homog(points):
  """ 点の集合（dim * n の配列）を同次座標系に変換する """

  return vstack((points,ones((1,points.shape[1]))))

def H_from_points(fp,tp):
  """ 線形なDLT法を使って fpをtpに対応づけるホモグラフィー行列Hを求める。
      点は自動的に調整される """

  if fp.shape != tp.shape:
    raise RuntimeError('number of points do not match')

  # 点を調整する（数値計算上重要）
  # 開始点
  m = mean(fp[:2], axis=1)
  maxstd = max(std(fp[:2], axis=1)) + 1e-9
  C1 = diag([1/maxstd, 1/maxstd, 1])
  C1[0][2] = -m[0]/maxstd
  C1[1][2] = -m[1]/maxstd
  fp = dot(C1,fp)

  # 対応点
  m = mean(tp[:2], axis=1)
  maxstd = max(std(tp[:2], axis=1)) + 1e-9
  C2 = diag([1/maxstd, 1/maxstd, 1])
  C2[0][2] = -m[0]/maxstd
  C2[1][2] = -m[1]/maxstd
  tp = dot(C2,tp)

  # 線形法のための行列を作る。対応ごとに2つの行になる。
  nbr_correspondences = fp.shape[1]
  A = zeros((2*nbr_correspondences,9))
  for i in range(nbr_correspondences):
    A[2*i] = [-fp[0][i],-fp[1][i],-1,0,0,0,
          tp[0][i]*fp[0][i],tp[0][i]*fp[1][i],tp[0][i]]
    A[2*i+1] = [0,0,0,-fp[0][i],-fp[1][i],-1,
          tp[1][i]*fp[0][i],tp[1][i]*fp[1][i],tp[1][i]]

  U,S,V = linalg.svd(A)
  H = V[8].reshape((3,3))

  # 調整を元に戻す
  H = dot(linalg.inv(C2),dot(H,C1))

  # 正規化して返す
  return H / H[2,2]

def Haffine_from_points(fp,tp):
  """ fpをtpに変換するアフィン変換行列Hを求める """

  if fp.shape != tp.shape:
    raise RuntimeError('number of points do not match')

  # 点を調整する
  # 開始点
  m = mean(fp[:2], axis=1)
  maxstd = max(std(fp[:2], axis=1)) + 1e-9
  C1 = diag([1/maxstd, 1/maxstd, 1])
  C1[0][2] = -m[0]/maxstd
  C1[1][2] = -m[1]/maxstd
  fp_cond = dot(C1,fp)

  # 対応点
  m = mean(tp[:2], axis=1)
  C2 = C1.copy()  # 2つの点群で、同じ拡大率を用いる
  C2[0][2] = -m[0]/maxstd
  C2[1][2] = -m[1]/maxstd
  tp_cond = dot(C2,tp)

  # 平均0になるよう調整する。平行移動はなくなる。
  A = concatenate((fp_cond[:2],tp_cond[:2]), axis=0)
  U,S,V = linalg.svd(A.T)

  # Hartley-Zisserman (第2版) p.130 に基づき行列B,Cを求める
  tmp = V[:2].T
  B = tmp[:2]
  C = tmp[2:4]

  tmp2 = concatenate((dot(C,linalg.pinv(B)),zeros((2,1))), axis=1)
  H = vstack((tmp2,[0,0,1]))

  # 調整を元に戻す
  H = dot(linalg.inv(C2),dot(H,C1))

  return H / H[2,2]

class RansacModel(object):
  """ http://www.scipy.org/Cookbook/RANSAC のransac.pyを用いて
    ホモグラフィーを当てはめるためのクラス """

  def __init__(self,debug=False):
    self.debug = debug

  def fit(self, data):
    """ 4つの対応点にホモグラフィーを当てはめる """

    # H_from_points() を当てはめるために転置する
    data = data.T

    # 元の点
    fp = data[:3,:4]
    # 対応点
    tp = data[3:,:4]

    # ホモグラフィーを当てはめて返す
    return H_from_points(fp,tp)

  def get_error( self, data, H):
    """ すべての対応にホモグラフィーを当てはめ、各変換点との誤差を返す。"""

    data = data.T

    # 元の点
    fp = data[:3]
    # 対応点
    tp = data[3:]

    # fpを変換
    fp_transformed = dot(H,fp)

    # 同次座標を正規化
#    zi = nonzero(abs(fp_transformed[2]) < 1e-15)
#    fp_transformed[0][zi] = 0.0
#    fp_transformed[1][zi] = 0.0
#    fp_transformed[2][zi] = 0.0
    nz = nonzero(fp_transformed[2])
    for i in range(3):
      fp_transformed[i][nz] /= fp_transformed[2][nz]
#      fp_transformed[i] /= fp_transformed[2]

    # 1点あたりの誤差を返す
    return sqrt( sum((tp-fp_transformed)**2,axis=0) )

def H_from_ransac(fp,tp,model,maxiter=1000,match_threshold=10):
  """ RANSACを用いて対応点からホモグラフィー行列Hをロバストに推定する
    (ransac.py は http://www.scipy.org/Cookbook/RANSAC を使用)

     入力: fp,tp (3*n 配列) 同次座標での点群 """

  import ransac

  # 対応点をグループ化する
  data = vstack((fp,tp))

  # Hを計算して返す
  H,ransac_data = ransac.ransac(data.T,model,4,maxiter,match_threshold,10,
                                return_all=True)
  return H,ransac_data['inliers']

