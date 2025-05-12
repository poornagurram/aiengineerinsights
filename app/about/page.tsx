import { Authors, allAuthors } from 'contentlayer/generated'
import { MDXLayoutRenderer } from 'pliny/mdx-components'
import AuthorLayout from '@/layouts/AuthorLayout'
import { coreContent } from 'pliny/utils/contentlayer'
import { genPageMetadata } from 'app/seo'

export const metadata = genPageMetadata({ title: 'About' })

export default function Page() {
  const defaultAuthor = allAuthors.find((p) => p.slug === 'default') as Authors
  const sparrowhawkAuthor = allAuthors.find((p) => p.slug === 'sparrowhawk') as Authors
  const defaultContent = coreContent(defaultAuthor)
  const sparrowhawkContent = coreContent(sparrowhawkAuthor)

  return (
    <div className="space-y-8">
      <AuthorLayout content={defaultContent}>
        <MDXLayoutRenderer code={defaultAuthor.body.code} />
      </AuthorLayout>
      <AuthorLayout content={sparrowhawkContent}>
        <MDXLayoutRenderer code={sparrowhawkAuthor.body.code} />
      </AuthorLayout>
    </div>
  )
}
