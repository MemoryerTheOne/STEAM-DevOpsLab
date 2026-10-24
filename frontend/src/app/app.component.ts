import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';

interface MetricCard {
  label: string;
  value: string;
  delta: string;
  trend: 'up' | 'down';
}

interface SequenceItem {
  id: number;
  name: string;
  category: string;
  type: 'keep' | 'delete' | 'insert';
}

interface Recommendation {
  rank: number;
  name: string;
  category: string;
  score: number;
  confidence: string;
}

interface CorrectionAction {
  kind: 'keep' | 'delete' | 'insert';
  label: string;
  item: string;
  note: string;
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  datasets = ['Beauty', 'Sports', 'Yelp'];
  selectedDataset = 'Beauty';

  metrics: MetricCard[] = [
    { label: 'Recall@5', value: '0.182', delta: '+8.4%', trend: 'up' },
    { label: 'Recall@10', value: '0.261', delta: '+6.9%', trend: 'up' },
    { label: 'Recall@20', value: '0.385', delta: '+4.8%', trend: 'up' },
    { label: 'MRR', value: '0.147', delta: '+7.1%', trend: 'up' }
  ];

  sequence: SequenceItem[] = [
    { id: 101, name: 'Moisturizer', category: 'Beauty', type: 'keep' },
    { id: 204, name: 'Facial Cleansing Brush', category: 'Beauty', type: 'keep' },
    { id: 332, name: 'Vitamin C Serum', category: 'Beauty', type: 'delete' },
    { id: 990, name: 'Sunscreen SPF 50', category: 'Beauty', type: 'keep' },
    { id: 120, name: 'Lip Balm', category: 'Beauty', type: 'insert' },
    { id: 441, name: 'Night Cream', category: 'Beauty', type: 'keep' }
  ];

  correctedActions: CorrectionAction[] = [
    { kind: 'delete', label: 'Removed', item: 'Vitamin C Serum', note: 'Likely accidental click after repeated product switching.' },
    { kind: 'insert', label: 'Inserted', item: 'Lip Balm', note: 'Missed item detected from adjacent activity pattern.' },
    { kind: 'keep', label: 'Kept', item: 'Sunscreen SPF 50', note: 'Strong behavioral alignment with prior sessions.' }
  ];

  recommendations: Recommendation[] = [
    { rank: 1, name: 'Hydrating Eye Cream', category: 'Skincare', score: 96, confidence: 'Very high' },
    { rank: 2, name: 'Niacinamide Serum', category: 'Skincare', score: 92, confidence: 'High' },
    { rank: 3, name: 'Cleanser Duo', category: 'Self-care', score: 88, confidence: 'High' },
    { rank: 4, name: 'Daily SPF Mist', category: 'Sun care', score: 81, confidence: 'Medium' },
    { rank: 5, name: 'Soft Cotton Pads', category: 'Beauty tools', score: 77, confidence: 'Medium' }
  ];

  performance = [
    { label: 'Keep ratio', value: 62, color: '#4fd1c5' },
    { label: 'Delete ratio', value: 18, color: '#f56565' },
    { label: 'Insert ratio', value: 20, color: '#8b5cf6' }
  ];

  insightSummary = 'The corrected sequence produces a stronger next-item intent than the raw interaction history by removing noise and recovering a probable missed purchase.';
}
