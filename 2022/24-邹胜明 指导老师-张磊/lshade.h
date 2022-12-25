#ifndef VINA_INDIVIDUAL_H
#define VINA_INDIVIDUAL_H

#include "model.h"
#include "conf.h"
#include "incrementable.h"
#include <algorithm>
#include "random.h"
#include "energy.h"


bool MyCompare(const output_type& x, const output_type& y);

struct ADE {
	sz num_saved_mins;//���out������
	int num_steps;//��������
	fl min_rmsd;//�������Ľ��
	ADE() : num_steps(3000),num_saved_mins(20), min_rmsd(1.0){ }
	void operator()(model& m, output_container& out, const precalculate& p, const igrid& ig, const precalculate& p_widened, const igrid& ig_widened, const vec& corner1, const vec& corner2, incrementable* increment_me, rng& generator) const;
};

struct ADE_aux {
	int po_size;//��Ⱥ��ģ
	int dim;// ά��
	int best_index; //��¼��ǰ��Ⱥ�������ŵĸ���
	fl c;  //���ϵ����0.2
	fl S_F;  //���浱ǰ��Ⱥ������Ÿ���F�� lehmer��ֵ
	fl S_CR;  //���浱ǰ��Ⱥ������Ÿ���CR�� ����ƽ����
	std::vector<output_type> nowpopulation;//��ʼ��Ⱥ
	std::vector<output_type> mutatepopulation;//���콻������Ⱥ
	ADE_aux() :  po_size(30), dim(7){}
	ADE_aux(const model& m) : po_size(30){
		dim = m.ligand_degrees_of_freedom(0) + 7;
		nowpopulation.resize(po_size);
		mutatepopulation.resize(po_size);
		S_F = 0.5;
		S_CR = 0.5;
		c = 0.2;
	}
	void mutate_cross(model& m, energy_cal& energy, rng& generator, const vec& corner1, const vec& corner2, const int& num_steps);
	void select(model& m, energy_cal& energy, rng& generator);

};








#endif
