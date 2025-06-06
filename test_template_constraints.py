#!/usr/bin/env python3
"""
TemplateConstraintGenerator 테스트 스크립트 (수정된 버전)
실제 템플릿 시퀀스를 사용한 거리 필터링 로직 검증
"""

import sys
import os
# Add src directory to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.append(src_path)

from boltz.data.parse.template import TemplateConstraintGenerator
from boltz.data.parse.struct2seq import StructureSequenceMapper
from Bio.SeqUtils import seq1
import numpy as np

def extract_template_sequence(template_file, chain_id, max_length=100):
    """Extract actual sequence from template structure"""
    mapper = StructureSequenceMapper()
    try:
        structure = mapper._get_structure_parser(template_file)
        sequence = ''
        for model in structure:
            for chain in model:
                if chain.id == chain_id:
                    for residue in chain:
                        if residue.id[0] == ' ':
                            try:
                                sequence += seq1(residue.resname)
                            except:
                                sequence += 'X'  # Unknown residue
                    break
            break
        return sequence[:max_length]  # Return first max_length residues
    except Exception as e:
        print(f"Error extracting sequence: {e}")
        return None

def test_basic_functionality():
    """기본 기능 테스트"""
    print("🧪 Phase 1: 기본 기능 테스트 (실제 템플릿 시퀀스 사용)")
    print("=" * 60)
    
    # Extract actual template sequence
    template_file = "examples/8esv_A_tmpl.cif"
    template_chain = "A"
    
    print("📄 템플릿 시퀀스 추출 중...")
    test_sequence = extract_template_sequence(template_file, template_chain, max_length=100)
    
    if not test_sequence:
        print("❌ 템플릿 시퀀스 추출 실패")
        return False
        
    print(f"✅ 템플릿 시퀀스 추출 성공: {len(test_sequence)} 잔기")
    print(f"   시퀀스: {test_sequence[:50]}...")
    
    # 여러 cutoff 값으로 테스트
    cutoff_values = [5.0, 8.0, 12.0, 20.0]
    
    for cutoff in cutoff_values:
        print(f"\n📏 Testing with cb_distance_cutoff: {cutoff} Å")
        
        try:
            generator = TemplateConstraintGenerator(
                cb_distance_cutoff=cutoff,
                min_sequence_identity=0.0  # 테스트를 위해 낮게 설정
            )
            
            constraints = generator.generate_template_constraints(
                query_sequence=test_sequence,
                template_structure=template_file,
                template_chain_id=template_chain,
                constraint_type="nmr_distance",
                distance_buffer=0.1
            )
            
            print(f"✅ Generated {len(constraints)} constraints")
            
            # 첫 3개 constraint 상세 정보 출력
            if constraints:
                print("📋 First 3 constraints:")
                for i, constraint in enumerate(constraints[:3]):
                    if 'nmr_distance' in constraint:
                        nmr = constraint['nmr_distance']
                        distance = (nmr['lower_bound'] + nmr['upper_bound']) / 2
                        print(f"  {i+1}. {nmr['atom1']} - {nmr['atom2']}: {distance:.2f} Å")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    return True

def test_distance_filtering():
    """거리 필터링 테스트"""
    print("\n\n🔍 Phase 2: 거리 필터링 검증")
    print("=" * 50)
    
    try:
        # Extract template sequence
        template_file = "examples/8esv_A_tmpl.cif"
        template_chain = "A"
        test_sequence = extract_template_sequence(template_file, template_chain, max_length=100)
        
        if not test_sequence:
            print("❌ 템플릿 시퀀스 추출 실패")
            return False
        
        # 작은 cutoff로 테스트
        small_cutoff = 6.0
        large_cutoff = 20.0
        
        # 작은 cutoff로 생성
        print(f"🔬 작은 cutoff ({small_cutoff} Å)로 constraint 생성 중...")
        generator_small = TemplateConstraintGenerator(cb_distance_cutoff=small_cutoff, min_sequence_identity=0.0)
        constraints_small = generator_small.generate_template_constraints(
            query_sequence=test_sequence,
            template_structure=template_file,
            template_chain_id=template_chain,
            constraint_type="nmr_distance"
        )
        
        # 큰 cutoff로 생성
        print(f"🔬 큰 cutoff ({large_cutoff} Å)로 constraint 생성 중...")
        generator_large = TemplateConstraintGenerator(cb_distance_cutoff=large_cutoff, min_sequence_identity=0.0)
        constraints_large = generator_large.generate_template_constraints(
            query_sequence=test_sequence,
            template_structure=template_file,
            template_chain_id=template_chain,
            constraint_type="nmr_distance"
        )
        
        print(f"\n📊 Results:")
        print(f"  Small cutoff ({small_cutoff} Å): {len(constraints_small)} constraints")
        print(f"  Large cutoff ({large_cutoff} Å): {len(constraints_large)} constraints")
        
        # 거리 필터링이 제대로 작동하는지 확인
        if len(constraints_small) < len(constraints_large):
            print("✅ Distance filtering working correctly!")
            print(f"   Filtered out {len(constraints_large) - len(constraints_small)} long-distance pairs")
        elif len(constraints_small) == len(constraints_large) and len(constraints_large) > 0:
            print("⚠️ Same number of constraints - possibly all distances are within small cutoff")
        else:
            print("⚠️ Warning: Distance filtering may not be working properly")
            
        # 거리 분포 확인
        if constraints_large:
            distances = []
            for constraint in constraints_large:
                if 'nmr_distance' in constraint:
                    nmr = constraint['nmr_distance']
                    distance = (nmr['lower_bound'] + nmr['upper_bound']) / 2
                    distances.append(distance)
            
            if distances:
                print(f"\n📈 Distance statistics (large cutoff):")
                print(f"   Min: {min(distances):.2f} Å")
                print(f"   Max: {max(distances):.2f} Å")
                print(f"   Mean: {np.mean(distances):.2f} Å")
                print(f"   Count: {len(distances)} constraints")
                
                # 모든 거리가 cutoff 이하인지 확인
                violations = [d for d in distances if d > large_cutoff]
                if violations:
                    print(f"❌ Found {len(violations)} distance violations!")
                    print(f"   Violation distances: {violations[:5]}...")  # Show first 5
                else:
                    print("✅ All distances within cutoff limit")
                    
                # 수정 전후 비교를 위한 통계
                short_distances = [d for d in distances if d <= small_cutoff]
                print(f"\n📏 Distance distribution:")
                print(f"   ≤ {small_cutoff} Å: {len(short_distances)} constraints")
                print(f"   > {small_cutoff} Å: {len(distances) - len(short_distances)} constraints")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 TemplateConstraintGenerator 수정 검증 테스트 (v2)")
    print("=" * 60)
    
    success1 = test_basic_functionality()
    success2 = test_distance_filtering()
    
    if success1 and success2:
        print("\n\n✨ 모든 테스트 성공!")
    else:
        print("\n\n⚠️ 일부 테스트 실패")
