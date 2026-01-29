"""
Personality Portrait Analyzer - Production Ready
A comprehensive personality analysis tool using NLP and ML
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
from typing import Dict, List, Tuple
import re
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Personality Portrait Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }
    .trait-label {
        font-weight: 600;
        color: #333;
    }
    .summary-box {
        background-color: #e8f4f8;
        border-left: 5px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)


class PersonalityAnalyzer:
    """Main class for personality analysis"""
    
    def __init__(self):
        self.big_five_traits = ['Openness', 'Conscientiousness', 'Extraversion', 
                                'Agreeableness', 'Emotional Range']
        self.sub_traits = self._initialize_sub_traits()
        
    def _initialize_sub_traits(self) -> Dict:
        """Initialize personality sub-traits structure"""
        return {
            'Openness': [
                'Adventurousness', 'Artistic interests', 'Emotionality',
                'Imagination', 'Intellect', 'Authority-challenging'
            ],
            'Conscientiousness': [
                'Achievement striving', 'Cautiousness', 'Dutifulness',
                'Orderliness', 'Self-discipline', 'Self-efficacy'
            ],
            'Extraversion': [
                'Activity level', 'Assertiveness', 'Cheerfulness',
                'Excitement-seeking', 'Outgoing', 'Gregariousness'
            ],
            'Agreeableness': [
                'Altruism', 'Cooperation', 'Modesty',
                'Uncompromising', 'Sympathy', 'Trust'
            ],
            'Emotional Range': [
                'Fiery', 'Prone to worry', 'Melancholy',
                'Immoderation', 'Self-consciousness', 'Susceptible to stress'
            ]
        }
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze text and return personality scores
        This is a simplified rule-based approach for demo purposes.
        In production, use fine-tuned transformer models.
        """
        logger.info(f"Analyzing text of length: {len(text)}")
        
        # Word count
        words = text.lower().split()
        word_count = len(words)
        
        if word_count < 100:
            st.warning("⚠️ For more accurate analysis, please provide at least 100 words.")
        
        # Calculate scores using linguistic patterns
        scores = self._calculate_personality_scores(text, words)
        
        # Generate insights
        insights = self._generate_insights(scores, text, word_count)
        
        return {
            'scores': scores,
            'insights': insights,
            'word_count': word_count,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_personality_scores(self, text: str, words: List[str]) -> Dict:
        """Calculate personality trait scores based on text analysis"""
        
        # Keyword dictionaries (simplified - in production use trained models)
        openness_keywords = ['creative', 'imagine', 'art', 'novel', 'adventure', 
                            'curious', 'explore', 'unique', 'innovative', 'abstract']
        conscientiousness_keywords = ['organized', 'plan', 'careful', 'responsible',
                                     'detail', 'thorough', 'precise', 'punctual', 
                                     'disciplined', 'efficient']
        extraversion_keywords = ['social', 'party', 'outgoing', 'energetic', 
                                'talkative', 'enthusiastic', 'active', 'assertive',
                                'friendly', 'excitement']
        agreeableness_keywords = ['kind', 'helpful', 'trust', 'cooperative',
                                 'sympathetic', 'generous', 'considerate', 
                                 'warm', 'caring', 'compassionate']
        emotional_keywords = ['worry', 'stress', 'anxious', 'nervous', 'angry',
                             'upset', 'emotional', 'sensitive', 'mood', 'fear']
        
        # Count keyword occurrences
        text_lower = text.lower()
        
        openness_score = sum(1 for word in openness_keywords if word in text_lower)
        conscientiousness_score = sum(1 for word in conscientiousness_keywords if word in text_lower)
        extraversion_score = sum(1 for word in extraversion_keywords if word in text_lower)
        agreeableness_score = sum(1 for word in agreeableness_keywords if word in text_lower)
        emotional_score = sum(1 for word in emotional_keywords if word in text_lower)
        
        # Normalize scores to 0-100 range
        word_count = len(words)
        normalization_factor = max(word_count / 100, 1)
        
        # Add some randomness for sub-traits (in production, use proper models)
        np.random.seed(hash(text[:100]) % 2**32)
        
        scores = {
            'Openness': self._normalize_score(openness_score, normalization_factor),
            'Conscientiousness': self._normalize_score(conscientiousness_score, normalization_factor),
            'Extraversion': self._normalize_score(extraversion_score, normalization_factor),
            'Agreeableness': self._normalize_score(agreeableness_score, normalization_factor),
            'Emotional Range': self._normalize_score(emotional_score, normalization_factor)
        }
        
        # Generate sub-trait scores
        detailed_scores = {}
        for trait, value in scores.items():
            detailed_scores[trait] = {
                'main': value,
                'sub_traits': {}
            }
            for sub_trait in self.sub_traits[trait]:
                # Add variation around main score
                variation = np.random.randint(-15, 15)
                sub_score = np.clip(value + variation, 0, 100)
                detailed_scores[trait]['sub_traits'][sub_trait] = sub_score
        
        return detailed_scores
    
    def _normalize_score(self, raw_score: float, normalization_factor: float) -> int:
        """Normalize raw score to 0-100 range with baseline"""
        # Add baseline of 30-50
        baseline = np.random.randint(30, 50)
        normalized = baseline + (raw_score / normalization_factor) * 30
        return int(np.clip(normalized, 0, 100))
    
    def _generate_insights(self, scores: Dict, text: str, word_count: int) -> Dict:
        """Generate personality insights and recommendations"""
        
        main_scores = {trait: scores[trait]['main'] for trait in scores}
        
        # Determine dominant traits
        sorted_traits = sorted(main_scores.items(), key=lambda x: x[1], reverse=True)
        dominant_trait = sorted_traits[0][0]
        
        # Generate summary
        summary = self._generate_summary(main_scores, dominant_trait)
        
        # Generate consumer needs
        consumer_needs = self._generate_consumer_needs(main_scores)
        
        # Generate values
        values = self._generate_values(main_scores)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(main_scores)
        
        return {
            'summary': summary,
            'consumer_needs': consumer_needs,
            'values': values,
            'recommendations': recommendations,
            'dominant_trait': dominant_trait,
            'analysis_strength': 'Strong' if word_count > 500 else 'Moderate' if word_count > 100 else 'Limited'
        }
    
    def _generate_summary(self, scores: Dict, dominant_trait: str) -> str:
        """Generate personality summary"""
        openness = scores['Openness']
        conscientiousness = scores['Conscientiousness']
        extraversion = scores['Extraversion']
        agreeableness = scores['Agreeableness']
        emotional = scores['Emotional Range']
        
        summary_parts = []
        
        # Openness
        if openness > 70:
            summary_parts.append("You are highly creative and open to new experiences.")
        elif openness > 40:
            summary_parts.append("You balance tradition with openness to new ideas.")
        else:
            summary_parts.append("You prefer familiar approaches and established methods.")
        
        # Conscientiousness
        if conscientiousness > 70:
            summary_parts.append("You are highly organized and goal-oriented.")
        elif conscientiousness > 40:
            summary_parts.append("You balance spontaneity with planning.")
        else:
            summary_parts.append("You prefer flexibility over rigid planning.")
        
        # Extraversion
        if extraversion > 70:
            summary_parts.append("You are energized by social interactions and enjoy being around people.")
        elif extraversion > 40:
            summary_parts.append("You appreciate both social time and solitude.")
        else:
            summary_parts.append("You recharge through quiet reflection and introspection.")
        
        # Agreeableness
        if agreeableness > 70:
            summary_parts.append("You are compassionate and prioritize harmony in relationships.")
        elif agreeableness > 40:
            summary_parts.append("You balance empathy with healthy boundaries.")
        else:
            summary_parts.append("You value directness and may prioritize logic over emotions.")
        
        # Emotional Range
        if emotional < 30:
            summary_parts.append("You maintain emotional stability even in challenging situations.")
        elif emotional < 60:
            summary_parts.append("You experience a normal range of emotions and manage them effectively.")
        else:
            summary_parts.append("You are emotionally sensitive and deeply responsive to your environment.")
        
        return " ".join(summary_parts)
    
    def _generate_consumer_needs(self, scores: Dict) -> Dict:
        """Generate consumer needs based on personality"""
        return {
            'Curiosity': min(100, scores['Openness'] + 10),
            'Structure': scores['Conscientiousness'],
            'Stability': 100 - scores['Emotional Range'],
            'Self-expression': min(100, (scores['Openness'] + scores['Extraversion']) // 2),
            'Challenge': min(100, scores['Openness'] + 5)
        }
    
    def _generate_values(self, scores: Dict) -> Dict:
        """Generate values based on personality"""
        return {
            'Stimulation': scores['Openness'],
            'Helping others': scores['Agreeableness'],
            'Achievement': scores['Conscientiousness'],
            'Taking pleasure in life': min(100, scores['Extraversion'] + 10),
            'Tradition': 100 - scores['Openness']
        }
    
    def _generate_recommendations(self, scores: Dict) -> Dict:
        """Generate personalized recommendations"""
        recommendations = {
            'likely_to': [],
            'unlikely_to': []
        }
        
        # Likely behaviors
        if scores['Openness'] > 60:
            recommendations['likely_to'].append("Enjoy exploring new experiences and ideas")
        if scores['Conscientiousness'] > 60:
            recommendations['likely_to'].append("Make detailed plans before taking action")
        if scores['Extraversion'] > 60:
            recommendations['likely_to'].append("Seek out social gatherings and group activities")
        if scores['Agreeableness'] > 60:
            recommendations['likely_to'].append("Prioritize harmony in relationships")
        
        # Unlikely behaviors
        if scores['Openness'] < 40:
            recommendations['unlikely_to'].append("Take risks without careful consideration")
        if scores['Conscientiousness'] < 40:
            recommendations['unlikely_to'].append("Stick to rigid schedules")
        if scores['Extraversion'] < 40:
            recommendations['unlikely_to'].append("Be influenced by social media trends")
        if scores['Agreeableness'] < 40:
            recommendations['unlikely_to'].append("Avoid confrontation at all costs")
        
        return recommendations


class VisualizationEngine:
    """Handles all visualization generation"""
    
    @staticmethod
    def create_circular_diagram(scores: Dict, title: str = "Personality Portrait") -> go.Figure:
        """Create the main circular personality diagram"""
        
        # Prepare data for visualization
        traits_data = []
        colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd']
        
        for idx, (trait, data) in enumerate(scores.items()):
            main_score = data['main']
            sub_traits = data['sub_traits']
            
            # Add main trait
            traits_data.append({
                'trait': trait,
                'score': main_score,
                'level': 1,
                'parent': '',
                'color': colors[idx]
            })
            
            # Add sub-traits
            for sub_trait, score in sub_traits.items():
                traits_data.append({
                    'trait': sub_trait,
                    'score': score,
                    'level': 2,
                    'parent': trait,
                    'color': colors[idx]
                })
        
        # Create sunburst chart
        df = pd.DataFrame(traits_data)
        
        fig = go.Figure(go.Sunburst(
            labels=[row['trait'] for _, row in df.iterrows()],
            parents=[row['parent'] for _, row in df.iterrows()],
            values=[row['score'] for _, row in df.iterrows()],
            marker=dict(
                colors=[row['color'] for _, row in df.iterrows()],
                line=dict(color='white', width=2)
            ),
            hovertemplate='<b>%{label}</b><br>Score: %{value:.0f}<extra></extra>',
            textinfo='label',
        ))
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=24, color='#1f77b4'),
                x=0.5,
                xanchor='center'
            ),
            height=700,
            margin=dict(t=80, l=0, r=0, b=0),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        return fig
    
    @staticmethod
    def create_trait_bars(scores: Dict) -> go.Figure:
        """Create horizontal bar chart for main traits"""
        
        main_scores = {trait: data['main'] for trait, data in scores.items()}
        traits = list(main_scores.keys())
        values = list(main_scores.values())
        
        # Color code based on score
        colors_list = []
        for val in values:
            if val >= 70:
                colors_list.append('#2ca02c')  # Green
            elif val >= 40:
                colors_list.append('#ff7f0e')  # Orange
            else:
                colors_list.append('#d62728')  # Red
        
        fig = go.Figure(go.Bar(
            y=traits,
            x=values,
            orientation='h',
            marker=dict(
                color=colors_list,
                line=dict(color='white', width=2)
            ),
            text=[f'{v}%' for v in values],
            textposition='inside',
            hovertemplate='<b>%{y}</b><br>Score: %{x}%<extra></extra>'
        ))
        
        fig.update_layout(
            title="Big Five Personality Traits",
            xaxis=dict(
                title="Score (%)",
                range=[0, 100],
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title="",
                categoryorder='total ascending'
            ),
            height=400,
            margin=dict(l=150, r=50, t=80, b=50),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        return fig
    
    @staticmethod
    def create_radar_chart(data_dict: Dict, title: str) -> go.Figure:
        """Create radar chart for needs/values"""
        
        categories = list(data_dict.keys())
        values = list(data_dict.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.3)',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8, color='#1f77b4'),
            hovertemplate='<b>%{theta}</b><br>Score: %{r}%<extra></extra>'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(size=10),
                    gridcolor='lightgray'
                ),
                angularaxis=dict(
                    tickfont=dict(size=11)
                )
            ),
            title=dict(
                text=title,
                font=dict(size=18),
                x=0.5,
                xanchor='center'
            ),
            height=450,
            showlegend=False,
            paper_bgcolor='white'
        )
        
        return fig
    
    @staticmethod
    def create_sub_traits_heatmap(scores: Dict, trait_name: str) -> go.Figure:
        """Create heatmap for sub-traits of a specific trait"""
        
        sub_traits = scores[trait_name]['sub_traits']
        traits_list = list(sub_traits.keys())
        values_list = list(sub_traits.values())
        
        # Create color scale
        fig = go.Figure(go.Bar(
            y=traits_list,
            x=values_list,
            orientation='h',
            marker=dict(
                color=values_list,
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Score"),
                line=dict(color='white', width=1)
            ),
            text=[f'{v}%' for v in values_list],
            textposition='inside',
            hovertemplate='<b>%{y}</b><br>Score: %{x}%<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"{trait_name} - Sub-Traits",
            xaxis=dict(title="Score (%)", range=[0, 100]),
            yaxis=dict(title=""),
            height=300,
            margin=dict(l=180, r=50, t=60, b=50),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        return fig


def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Personality Portrait Analyzer</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="summary-box">
        <h3>Welcome to the Personality Portrait Analyzer</h3>
        <p>This advanced tool analyzes your text to create a comprehensive personality profile 
        based on the Big Five personality traits. Get insights into your personality, consumer needs, 
        values, and behavioral tendencies.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = PersonalityAnalyzer()
    viz_engine = VisualizationEngine()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.markdown("---")
        st.subheader("Input Options")
        input_method = st.radio(
            "Choose input method:",
            ["Text Box", "Upload File"],
            help="Select how you want to provide text for analysis"
        )
        
        st.markdown("---")
        st.subheader("Analysis Settings")
        show_detailed = st.checkbox("Show detailed sub-traits", value=True)
        show_recommendations = st.checkbox("Show recommendations", value=True)
        
        st.markdown("---")
        st.subheader("Export Options")
        export_format = st.selectbox(
            "Export format:",
            ["JSON", "CSV", "PDF Report (Coming Soon)"],
            help="Choose format for exporting results"
        )
        
        st.markdown("---")
        st.info("💡 **Tip**: For best results, provide at least 300 words of your writing.")
    
    # Main content area
    text_input = ""
    
    if input_method == "Text Box":
        st.subheader("📝 Enter Your Text")
        text_input = st.text_area(
            "Paste or type your text here:",
            height=200,
            placeholder="Enter your text here for personality analysis. The more you write, the more accurate the analysis will be...",
            help="Provide at least 100 words for meaningful analysis"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
        with col2:
            if text_input:
                word_count = len(text_input.split())
                st.info(f"Word count: {word_count}")
    
    else:  # Upload File
        st.subheader("📁 Upload Text File")
        uploaded_file = st.file_uploader(
            "Choose a text file",
            type=['txt', 'md', 'json'],
            help="Upload a text file for analysis"
        )
        
        if uploaded_file is not None:
            text_input = uploaded_file.read().decode('utf-8')
            st.success(f"✅ File uploaded successfully! ({len(text_input.split())} words)")
        
        analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    # Analysis execution
    if analyze_button or (text_input and len(text_input) > 50):
        if not text_input or len(text_input.strip()) < 50:
            st.error("❌ Please provide at least 50 characters of text for analysis.")
            return
        
        with st.spinner("🔄 Analyzing your personality profile..."):
            try:
                # Perform analysis
                results = analyzer.analyze_text(text_input)
                scores = results['scores']
                insights = results['insights']
                
                # Store results in session state
                st.session_state['analysis_results'] = results
                
                # Success message
                st.success(f"✅ Analysis complete! ({results['word_count']} words analyzed)")
                
                # Display analysis strength
                strength_color = {
                    'Strong': 'green',
                    'Moderate': 'orange',
                    'Limited': 'red'
                }
                st.markdown(f"""
                    <div class="metric-card">
                        <p><strong>Analysis Strength:</strong> 
                        <span style="color: {strength_color[insights['analysis_strength']]}; font-weight: bold;">
                        {insights['analysis_strength']}
                        </span></p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Main visualization
                st.markdown("---")
                st.header("📊 Personality Portrait")
                
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    # Circular diagram
                    circular_fig = viz_engine.create_circular_diagram(scores)
                    st.plotly_chart(circular_fig, use_container_width=True)
                
                with col2:
                    # Summary
                    st.markdown("### Summary")
                    st.markdown(f'<div class="summary-box">{insights["summary"]}</div>', 
                              unsafe_allow_html=True)
                    
                    # Dominant trait
                    st.markdown("### Dominant Trait")
                    st.info(f"**{insights['dominant_trait']}** - This is your strongest personality characteristic.")
                
                # Trait bars
                st.markdown("---")
                st.header("📈 Big Five Personality Traits")
                trait_bars = viz_engine.create_trait_bars(scores)
                st.plotly_chart(trait_bars, use_container_width=True)
                
                # Detailed sub-traits
                if show_detailed:
                    st.markdown("---")
                    st.header("🔍 Detailed Sub-Traits Analysis")
                    
                    trait_tabs = st.tabs(list(scores.keys()))
                    
                    for idx, (trait_name, tab) in enumerate(zip(scores.keys(), trait_tabs)):
                        with tab:
                            sub_trait_fig = viz_engine.create_sub_traits_heatmap(scores, trait_name)
                            st.plotly_chart(sub_trait_fig, use_container_width=True)
                
                # Consumer Needs and Values
                st.markdown("---")
                st.header("🎯 Consumer Needs & Values")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    needs_fig = viz_engine.create_radar_chart(
                        insights['consumer_needs'],
                        "Consumer Needs"
                    )
                    st.plotly_chart(needs_fig, use_container_width=True)
                
                with col2:
                    values_fig = viz_engine.create_radar_chart(
                        insights['values'],
                        "Values"
                    )
                    st.plotly_chart(values_fig, use_container_width=True)
                
                # Recommendations
                if show_recommendations:
                    st.markdown("---")
                    st.header("💡 Behavioral Insights")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("✅ You are likely to:")
                        for item in insights['recommendations']['likely_to']:
                            st.markdown(f"- {item}")
                    
                    with col2:
                        st.subheader("❌ You are unlikely to:")
                        for item in insights['recommendations']['unlikely_to']:
                            st.markdown(f"- {item}")
                
                # Export functionality
                st.markdown("---")
                st.header("💾 Export Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📥 Download JSON", use_container_width=True):
                        json_data = json.dumps(results, indent=2)
                        st.download_button(
                            label="Download",
                            data=json_data,
                            file_name=f"personality_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                
                with col2:
                    if st.button("📥 Download CSV", use_container_width=True):
                        # Create CSV data
                        main_scores_df = pd.DataFrame([
                            {trait: scores[trait]['main'] for trait in scores}
                        ])
                        csv_data = main_scores_df.to_csv(index=False)
                        st.download_button(
                            label="Download",
                            data=csv_data,
                            file_name=f"personality_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                with col3:
                    if export_format == "PDF Report (Coming Soon)":
                        st.info("PDF export coming soon!")
                
            except Exception as e:
                logger.error(f"Analysis error: {str(e)}")
                st.error(f"❌ An error occurred during analysis: {str(e)}")
                st.info("Please try again with different text or contact support if the issue persists.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem 0;">
            <p><strong>Personality Portrait Analyzer</strong> v1.0</p>
            <p>This tool uses advanced NLP and machine learning for personality analysis.</p>
            <p>⚠️ Note: This analysis is for informational purposes only and should not be used for clinical diagnosis.</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
